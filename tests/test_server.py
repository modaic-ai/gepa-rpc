from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Optional, Mapping, Sequence
import uvicorn
from .classifier import InstructorAdapter, metric
from dotenv import load_dotenv
from gepa_rpc.models import TraceData, Prediction, ReflectiveExample
from gepa import EvaluationBatch

load_dotenv()

app = FastAPI()
with open("tests/labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]
adapter = InstructorAdapter(
    starter_prompt=f"Classify the following support ticket. Allowed categories: {', '.join(labels)}"
)


class EvaluateRequest(BaseModel):
    batch: list[dict[str, Any]]
    candidate: dict[str, str]
    capture_traces: bool = False


class MakeReflectiveDatasetRequest(BaseModel):
    candidate: dict[str, str]
    eval_batch: EvaluationBatch[TraceData, Prediction]
    components_to_update: list[str]


# --- GEPA Endpoints ---


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest) -> EvaluationBatch:
    """
    Evaluates the Instructor classification agent.
    - request.candidate['classifier_prompt'] is the system prompt to optimize.
    - request.batch is a list of support tickets to classify.
    """
    # Update the adapter's system prompt to the current candidate
    adapter.system_prompt = request.candidate.get(
        "classifier_prompt", adapter.system_prompt
    )

    outputs = []
    scores = []
    trajectories = []

    for item in request.batch:
        ticket_text = item.get("text", "")
        expected_category = item.get("label_text", "")  # For scoring

        # Run the classification via the adapter's __call__
        result_dict = adapter(ticket_text)

        # Scoring: 1.0 if category matches expectation
        # result_dict matches the ClassificationResult schema
        score = (
            1.0
            if result_dict.get("category", "").lower() == expected_category.lower()
            else 0.0
        )

        outputs.append(result_dict)
        scores.append(score)

        if request.capture_traces:
            # We construct a minimal trajectory since __call__ returns only the result dict
            trajectories.append({"input": ticket_text, "output": result_dict})

    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories if request.capture_traces else None,
    )


@app.post("/make_reflective_dataset")
async def make_reflective_dataset(
    request: MakeReflectiveDatasetRequest,
) -> dict[str, list[ReflectiveExample]]:
    """
    Constructs feedback for GEPA to improve the classification system prompt.
    """
    dataset: dict[str, list[ReflectiveExample]] = {}

    for component in request.components_to_update:
        items = []
        for i, score in enumerate(request.eval_batch.scores):
            # Simple feedback logic
            if score < 1.0:
                feedback = "The agent misclassified this ticket."
            else:
                feedback = "Perfect classification."

            metric_result = metric(
                example=request.eval_batch.trajectories[i].example,
                prediction=request.eval_batch.outputs[i],
                trace=request.eval_batch.trajectories[i].trace,
                pred_name=component,
                pred_trace=[
                    t
                    for t in request.eval_batch.trajectories[i].trace
                    if t[0] == component
                ],
            )

            items.append(
                ReflectiveExample(
                    Inputs=request.eval_batch.trajectories[i]["input"]
                    if request.eval_batch.trajectories
                    else {"text": "unknown"},
                    Generated_Outputs=request.eval_batch.outputs[i],
                    Feedback=feedback,
                )
            )
        dataset[component] = items

    return dataset


if __name__ == "__main__":
    print("Starting LangChain Semantic Classification GEPA RPC Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
