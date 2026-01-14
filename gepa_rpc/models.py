from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict
from gepa import EvaluationBatch

Example = dict[str, Any]

## Core Cross Language Models ##


class Prediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    output: Any | None
    error_type: str | None = Field(default=None, alias="errorType")
    error_message: str | None = Field(default=None, alias="errorMessage")
    error_traceback: str | None = Field(default=None, alias="errorTraceback")


class TraceEntry(BaseModel):
    predictor: str
    input: Any
    output: Any


class ScoreWithFeedback(BaseModel):
    score: float
    feedback: str | None


class Trace(BaseModel):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[TraceEntry]
    score: ScoreWithFeedback


################################

## GEPA Adapter Models ##


class ReflectiveExample(TypedDict):
    Inputs: Example  # Predictor inputs (may include str, dspy.Image, etc.)
    Generated_Outputs: (
        Prediction  # Success: dict with output fields, Failure: error message string
    )
    Feedback: str  # Always a string - from metric function or parsing error message


## Request Models ##


class EvaluateRequest(BaseModel):
    batch: list[Example]
    candidate: dict[str, str]
    capture_traces: bool = False


class MakeReflectiveDatasetRequest(BaseModel):
    candidate: dict[str, str]
    eval_batch: EvaluationBatch[Trace, Prediction]
    components_to_update: list[str]
