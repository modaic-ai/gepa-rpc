from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Optional, Mapping, Sequence
import uvicorn
from .langchain import LangchainAdapter
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
with open("tests/labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]
adapter = LangchainAdapter(
    starter_prompt=f"Classify the following support ticket. Allowed categories: {', '.join(labels)}"
)

# --- GEPA RPC Models ---


class EvaluationBatchModel(BaseModel):
    outputs: list[Any]
    scores: list[float]
    trajectories: Optional[list[Any]] = None


class EvaluateRequest(BaseModel):
    batch: list[dict[str, Any]]
    candidate: dict[str, str]
    capture_traces: bool = False


class ReflectiveExample(BaseModel):
    Inputs: dict[str, Any]
    Generated_Outputs: Any = Field(alias="Generated Outputs")
    Feedback: str

    model_config = {"populate_by_name": True}


class MakeReflectiveDatasetRequest(BaseModel):
    candidate: dict[str, str]
    eval_batch: EvaluationBatchModel
    components_to_update: list[str]


# --- GEPA Endpoints ---


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest) -> EvaluationBatchModel:
    """
    Evaluates the LangChain classification agent.
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
        expected_category = item.get("label", "")  # For scoring

        # Run the classification via the adapter's __call__
        result_dict = await adapter(ticket_text)

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

    return EvaluationBatchModel(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories if request.capture_traces else None,
    )


@app.post("/make_reflective_dataset")
async def make_reflective_dataset(
    request: MakeReflectiveDatasetRequest,
) -> Mapping[str, Sequence[ReflectiveExample]]:
    """
    Constructs feedback for GEPA to improve the classification system prompt.
    """
    dataset: dict[str, list[ReflectiveExample]] = {}

    for component in request.components_to_update:
        if component == "classifier_prompt":
            items = []
            for i, score in enumerate(request.eval_batch.scores):
                # Simple feedback logic
                if score < 1.0:
                    feedback = "The agent misclassified this ticket. It should have been more attentive to keywords related to the category."
                else:
                    feedback = "Perfect classification."

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
