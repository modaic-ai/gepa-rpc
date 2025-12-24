import os
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from typing import Literal
from langchain.agents.middleware import dynamic_prompt, ModelRequest

with open("tests/labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]


# Define the structured output schema
class ClassificationResult(BaseModel):
    category: Literal[labels] = Field(
        description="The category that best describes the input text."
    )


class Context(BaseModel):
    system_prompt: str


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    system_prompt = request.runtime.context.system_prompt
    return system_prompt


class LangchainAdapter:
    def __init__(self, starter_prompt: str, model_name: str = "gpt-4o-mini"):
        """
        Initializes the LangchainAdapter.

        Args:
            starter_prompt: The initial system prompt to guide the classification.
            model_name: The name of the model to use (default: gpt-4o-mini).
        """
        self.starter_prompt = starter_prompt
        self.system_prompt = starter_prompt
        self.model_name = model_name
        self.agent = create_agent(
            model=self.model_name,
            system_prompt=self.system_prompt,
            response_format=ToolStrategy(ClassificationResult),
            middleware=[dynamic_system_prompt],
            context_schema=Context,
        )

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
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            context={"system_prompt": self.system_prompt},
        )
        print("result", result)

        # Return the dumped base model as a dict
        if "structured_response" in result:
            return result["structured_response"].model_dump()

        # Fallback if the expected key is missing
        return {
            "category": "error",
        }

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt


def metric(
    example: dict[str, Any], prediction: dict[str, Any], trace: list[dict[str, Any]]
) -> float:
    return 1.0 if prediction["category"] == example["label_text"] else 0.0


if __name__ == "__main__":
    from dotenv import load_dotenv

    with open("tests/labels.txt", "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    load_dotenv()
    adapter = LangchainAdapter(
        starter_prompt=f"Classify the following support ticket. Allowed categories: {', '.join(labels)}"
    )
    result = adapter("I need help with my account")
    print("result", result)
