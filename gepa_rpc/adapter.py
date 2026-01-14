# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa_rpc.models import Prediction


import requests

from gepa.core.adapter import GEPAAdapter, EvaluationBatch

from .models import Example, Trace, Prediction, ReflectiveExample


class RPCAdapter(GEPAAdapter[Example, Trace, Prediction]):
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
    ) -> EvaluationBatch[Trace, Prediction]:
        """
        Forward evaluation request to the remote server.
        """
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

        return EvaluationBatch[Trace, Prediction](**data)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trace, Prediction],
        components_to_update: list[str],
    ) -> dict[str, list[ReflectiveExample]]:
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
