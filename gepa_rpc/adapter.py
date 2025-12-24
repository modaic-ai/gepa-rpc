# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import requests
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Protocol, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

# DataInst, Trajectory, RolloutOutput
Example = dict[str, Any]
Prediction = dict[str, Any]


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


class ReflectiveExample(TypedDict):
    """
    Structure of individual examples in the reflective dataset.

    Each example contains the predictor inputs, generated outputs, and feedback from evaluation.
    """

    Inputs: dict[str, Any]  # Predictor inputs (may include str, dspy.Image, etc.)
    Generated_Outputs: (
        dict[str, Any] | str
    )  # Success: dict with output fields, Failure: error message string
    Feedback: str  # Always a string - from metric function or parsing error message


class ScoreWithFeedback(TypedDict):
    score: float
    feedback: Optional[str]


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


class RPCAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    """
    A GEPA Adapter that forwards calls to a remote server (e.g., a TypeScript server)
    via HTTP. This allows you to implement your system logic in any language.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TraceData, Prediction]:
        """
        Forward evaluation request to the remote server.
        """
        print("batch", batch)
        print("candidate", candidate)
        print("capture_traces", capture_traces)
        response = requests.post(
            f"{self.base_url}/evaluate",
            json={
                "batch": batch,
                "candidate": candidate,
                "capture_traces": capture_traces,
            },
        )
        response.raise_for_status()
        data = response.json()

        return EvaluationBatch(
            outputs=data["outputs"],
            scores=data["scores"],
            trajectories=data.get("trajectories"),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TraceData, Prediction],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Forward reflective dataset construction request to the remote server.
        """
        # Convert EvaluationBatch to a dict for JSON serialization
        eval_batch_dict = {
            "outputs": eval_batch.outputs,
            "scores": eval_batch.scores,
            "trajectories": eval_batch.trajectories,
        }

        response = requests.post(
            f"{self.base_url}/make_reflective_dataset",
            json={
                "candidate": candidate,
                "eval_batch": eval_batch_dict,
                "components_to_update": components_to_update,
            },
        )
        response.raise_for_status()
        return response.json()
