from typing import Any, Dict, List
from pydantic import BaseModel, Field
from typing import Literal
import instructor
from gepa_rpc.models import Example, Prediction, Trace, ScoreWithFeedback
from typing import Optional

with open("tests/labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]


# Define the structured output schema
class ClassificationResult(BaseModel):
    category: Literal[*labels] = Field(
        description="The category that best describes the input text."
    )


class InstructorAdapter:
    def __init__(self, starter_prompt: str, model_name: str = "openai/gpt-4o-mini"):
        """
        Initializes the LangchainAdapter.

        Args:
            starter_prompt: The initial system prompt to guide the classification.
            model_name: The name of the model to use (default: gpt-4o-mini).
        """
        self.starter_prompt = starter_prompt
        self.system_prompt = starter_prompt
        self.model_name = model_name
        self.client = instructor.from_provider(self.model_name)

    def __call__(self, text: str) -> Dict[str, Any]:
        """
        Runs the LangChain adapter to classify the input text.

        Args:
            text: The input text to classify.

        Returns:
            A dictionary containing the structured classification result.
        """
        # Assuming invoke might be async or sync depending on implementation;
        # using a common pattern for async agents if needed.
        result = self.client.create(
            response_model=ClassificationResult,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        return result.model_dump()

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt


def metric(
    example: str,
    prediction: str,
    trace: Optional[Trace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Trace] = None,
) -> float | ScoreWithFeedback:
    score = 1.0 if prediction["category"] == example["label_text"] else 0.0
    if trace and pred_name and pred_trace:
        if prediction["category"] not in labels:
            return ScoreWithFeedback(
                score=0.0,
                feedback=f"Prediction category {prediction['category']} not a valid label",
            )
    return score


if __name__ == "__main__":
    from dotenv import load_dotenv

    with open("tests/labels.txt", "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    load_dotenv()
    adapter = InstructorAdapter(
        starter_prompt=f"Classify the following support ticket. Allowed categories: {', '.join(labels)}"
    )
    result = adapter("I need help with my account")
    print("result", result)
