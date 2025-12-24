# GEPA RPC Plan

I would like to run GEPA in typescript by basically by exposing my typescript code as a server and running gepa as a client. Which makes two different HTTP request to the server.

- evaluate
- make_reflective_dataset

These endpoints mirror the two main callback functions implemented for the GEPA Adapter Protocol.

## Examples

The definiton of the GEPA Adapter Protocol. (this is what you give to the gepa library to have it optimize your system)

```python
# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

# Generic type aliases matching your original
RolloutOutput = TypeVar("RolloutOutput")
Trajectory = TypeVar("Trajectory")
DataInst = TypeVar("DataInst")
Candidate = dict[str, str]
EvaluatorFn = Callable[[list[DataInst], Candidate], tuple[list[RolloutOutput], list[float]]]


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """
    Container for the result of evaluating a proposed candidate on a batch of data.

    - outputs: raw per-example outputs from upon executing the candidate. GEPA does not interpret these;
      they are forwarded to other parts of the user's code or logging as-is.
    - scores: per-example numeric scores (floats). GEPA sums these for minibatch acceptance
      and averages them over the full validation set for tracking/pareto fronts.
    - trajectories: optional per-example traces used by make_reflective_dataset to build
      a reflective dataset (See `GEPAAdapter.make_reflective_dataset`). If capture_traces=True is passed to `evaluate`, trajectories
      should be provided and align one-to-one with `outputs` and `scores`.
    """

    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None


class ProposalFn(Protocol):
    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        - Given the current `candidate`, a reflective dataset (as returned by
          `GEPAAdapter.make_reflective_dataset`), and a list of component names to update,
          return a mapping component_name -> new component text (str). This allows the user
          to implement their own instruction proposal logic. For example, the user can use
          a different LLM, implement DSPy signatures, etc. Another example can be situations
          where 2 or more components need to be updated together (coupled updates).

        Returns
        - Dict[str, str] mapping component names to newly proposed component texts.
        """
        ...


class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    """
    GEPAAdapter is the single integration point between your system
    and the GEPA optimization engine. Implementers provide three responsibilities:

    The following are user-defined types that are not interpreted by GEPA but are used by the user's code
        to define the adapter:
    DataInst: User-defined type of input data to the program under optimization.
    Trajectory: User-defined type of trajectory data, which typically captures the
        different steps of the program candidate execution.
    RolloutOutput: User-defined type of output data from the program candidate.

    The following are the responsibilities of the adapter:
    1) Program construction and evaluation (evaluate):
       Given a batch of DataInst and a "candidate" program (mapping from named components
       -> component text), execute the program to produce per-example scores and
       optionally rich trajectories (capturing intermediate states) needed for reflection.

    2) Reflective dataset construction (make_reflective_dataset):
       Given the candidate, EvaluationBatch (trajectories, outputs, scores), and the list of components to update,
       produce a small JSON-serializable dataset for each component that you want to update. This
       dataset is fed to the teacher LM to propose improved component text.

    3) Optional instruction proposal (propose_new_texts):
       GEPA provides a default implementation (instruction_proposal.py) that serializes the reflective dataset
       to propose new component texts. However, users can implement their own proposal logic by implementing this method.
       This method receives the current candidate, the reflective dataset, and the list of components to update,
       and returns a mapping from component name to new component text.

    Key concepts and contracts:
    - candidate: Dict[str, str] mapping a named component of the system to its corresponding text.
    - scores: higher is better. GEPA uses:
      - minibatch: sum(scores) to compare old vs. new candidate (acceptance test),
      - full valset: mean(scores) for tracking and Pareto-front selection.
      Ensure your metric is calibrated accordingly or normalized to a consistent scale.
    - trajectories: opaque to GEPA (the engine never inspects them). They must be
      consumable by your own make_reflective_dataset implementation to extract the
      minimal context needed to produce meaningful feedback for every component of
      the system under optimization.
    - error handling: Never raise for individual example failures. Instead:
      - Return a valid `EvaluationBatch` with per-example failure scores (e.g., 0.0)
        when formatting/parsing fails. Even better if the trajectories are also populated
        with the failed example, including the error message, identifying the reason for the failure.
      - Reserve exceptions for unrecoverable, systemic failures (e.g., missing model,
        misconfigured program, schema mismatch).
      - If an exception is raised, the engine will log the error and proceed to the next iteration.
    """

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Run the program defined by `candidate` on a batch of data.

        Parameters
        - batch: list of task-specific inputs (DataInst).
        - candidate: mapping from component name -> component text. You must instantiate
          your full system with the component text for each component, and execute it on the batch.
        - capture_traces: when True, you must populate `EvaluationBatch.trajectories`
          with a per-example trajectory object that your `make_reflective_dataset` can
          later consume. When False, you may set trajectories=None to save time/memory.
          capture_traces=True is used by the reflective mutation proposer to build a reflective dataset.

        Returns
        - EvaluationBatch with:
          - outputs: raw per-example outputs (opaque to GEPA).
          - scores: per-example floats, length == len(batch). Higher is better.
          - trajectories:
              - if capture_traces=True: list[Trajectory] with length == len(batch).
              - if capture_traces=False: None.

        Scoring semantics
        - The engine uses sum(scores) on minibatches to decide whether to accept a
          candidate mutation and average(scores) over the full valset for tracking.
        - Prefer to return per-example scores, that can be aggregated via summation.
        - If an example fails (e.g., parse error), use a fallback score (e.g., 0.0).

        Correctness constraints
        - len(outputs) == len(scores) == len(batch)
        - If capture_traces=True: trajectories must be provided and len(trajectories) == len(batch)
        - Do not mutate `batch` or `candidate` in-place. Construct a fresh program
          instance or deep-copy as needed.
        """
        ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build a small, JSON-serializable dataset (per component) to drive instruction
        refinement by a teacher LLM.

        Parameters
        - candidate: the same candidate evaluated in evaluate().
        - eval_batch: The result of evaluate(..., capture_traces=True) on
          the same batch. You should extract everything you need from eval_batch.trajectories
          (and optionally outputs/scores) to assemble concise, high-signal examples.
        - components_to_update: subset of component names for which the proposer has
          requested updates. At a time, GEPA identifies a subset of components to update.

        Returns
        - A dict: component_name -> list of dict records (the "reflective dataset").
          Each record should be JSON-serializable and is passed verbatim to the
          instruction proposal prompt. A recommended schema is:
            {
              "Inputs": Dict[str, str],             # Minimal, clean view of the inputs to the component
              "Generated Outputs": Dict[str, str] | str,  # Model outputs or raw text
              "Feedback": str                       # Feedback on the component's performance, including correct answer, error messages, etc.
            }
          You may include additional keys (e.g., "score", "rationale", "trace_id") if useful.

        Determinism
        - If you subsample trace instances, use a seeded RNG to keep runs reproducible.
        """
        ...

    propose_new_texts: ProposalFn | None = None

```

How DSPy implements the GEPA Adapter Protocol. I would like to use a similar data model as DSPy with out all the added DSPy specific features and complexity.

````python
import logging
import random
from typing import Any, Callable, Protocol, TypedDict

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.adapter import ProposalFn

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import History
from dspy.adapters.types.base_type import Type
from dspy.evaluate import Evaluate
from dspy.primitives import Example, Prediction
from dspy.teleprompt.bootstrap_trace import TraceData

logger = logging.getLogger(__name__)

class LoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log(self, x: str):
        self.logger.info(x)

DSPyTrace = list[tuple[Any, dict[str, Any], Prediction]]


class ReflectiveExample(TypedDict):
    """
    Structure of individual examples in the reflective dataset.

    Each example contains the predictor inputs, generated outputs, and feedback from evaluation.
    """
    Inputs: dict[str, Any]                              # Predictor inputs (may include str, dspy.Image, etc.)
    Generated_Outputs: dict[str, Any] | str             # Success: dict with output fields, Failure: error message string
    Feedback: str                                       # Always a string - from metric function or parsing error message


class ScoreWithFeedback(Prediction):
    score: float
    feedback: str

class PredictorFeedbackFn(Protocol):
    def __call__(
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Example,
        module_outputs: Prediction,
        captured_trace: DSPyTrace,
    ) -> ScoreWithFeedback:
        """
        This function is used to provide feedback to a specific predictor.
        The function is called with the following arguments:
        - predictor_output: The output of the predictor.
        - predictor_inputs: The inputs to the predictor.
        - module_inputs: The inputs to the whole program --- `Example`.
        - module_outputs: The outputs of the whole program --- `Prediction`.
        - captured_trace: The trace of the module's execution.
        # Shape of trace is: [predictor_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)

        The function should return a `ScoreWithFeedback` object.
        The feedback is a string that is used to guide the evolution of the predictor.
        """
        ...

class DspyAdapter(GEPAAdapter[Example, TraceData, Prediction]):
    def __init__(
        self,
        student_module,
        metric_fn: Callable,
        feedback_map: dict[str, Callable],
        failure_score=0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
        reflection_lm=None,
        custom_instruction_proposer: "ProposalFn | None" = None,
        warn_on_score_mismatch: bool = True
    ):
        self.student = student_module
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)
        self.reflection_lm = reflection_lm
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch

        if self.custom_instruction_proposer is not None:
            # We are only overriding the propose_new_texts method when a custom
            # instruction proposer is provided. Otherwise, we use the GEPA
            # default propose_new_texts.

            def custom_propose_new_texts(
                candidate: dict[str, str],
                reflective_dataset: dict[str, list[dict[str, Any]]],
                components_to_update: list[str]
            ) -> dict[str, str]:
                if self.reflection_lm is not None:
                    with dspy.context(lm=self.reflection_lm):
                        return self.custom_instruction_proposer(
                            candidate=candidate,
                            reflective_dataset=reflective_dataset,
                            components_to_update=components_to_update
                        )
                else:
                    return self.custom_instruction_proposer(
                        candidate=candidate,
                        reflective_dataset=reflective_dataset,
                        components_to_update=components_to_update
                    )

            self.propose_new_texts = custom_propose_new_texts

        # Cache predictor names/signatures
        self.named_predictors = list(self.student.named_predictors())


    def build_program(self, candidate: dict[str, str]):
        new_prog = self.student.deepcopy()
        for name, pred in new_prog.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])
        return new_prog

    def evaluate(self, batch, candidate, capture_traces=False):
        program = self.build_program(candidate)

        if capture_traces:
            # bootstrap_trace_data-like flow with trace capture
            from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module

            eval_callback_metadata = {"disable_logging": True}
            trajs = bootstrap_trace_module.bootstrap_trace_data(
                program=program,
                dataset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
                callback_metadata=eval_callback_metadata,
            )
            scores = []
            outputs = []
            for t in trajs:
                outputs.append(t["prediction"])
                if hasattr(t["prediction"], "__class__") and t.get("score") is None:
                    scores.append(self.failure_score)
                else:
                    score = t["score"]
                    if hasattr(score, "score"):
                        score = score["score"]
                    scores.append(score)
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajs)
        else:
            evaluator = Evaluate(
                devset=batch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(batch) * 100
            )
            res = evaluator(program)
            outputs = [r[1] for r in res.results]
            scores = [r[2] for r in res.results]
            scores = [s["score"] if hasattr(s, "score") else s for s in scores]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update) -> dict[str, list[ReflectiveExample]]:
        from dspy.teleprompt.bootstrap_trace import FailedPrediction
        program = self.build_program(candidate)

        ret_d: dict[str, list[ReflectiveExample]] = {}
        for pred_name in components_to_update:
            module = None
            for name, m in program.named_predictors():
                if name == pred_name:
                    module = m
                    break
            assert module is not None

            items: list[ReflectiveExample] = []
            for data in eval_batch.trajectories or []:
                trace = data["trace"]
                example = data["example"]
                prediction = data["prediction"]
                module_score = data["score"]
                if hasattr(module_score, "score"):
                    module_score = module_score["score"]

                trace_instances = [t for t in trace if t[0].signature.equals(module.signature)]
                if not self.add_format_failure_as_feedback:
                    trace_instances = [t for t in trace_instances if not isinstance(t[2], FailedPrediction)]
                if len(trace_instances) == 0:
                    continue

                selected = None
                for t in trace_instances:
                    if isinstance(t[2], FailedPrediction):
                        selected = t
                        break

                if selected is None:
                    if isinstance(prediction, FailedPrediction):
                        continue
                    selected = self.rng.choice(trace_instances)

                inputs = selected[1]
                outputs = selected[2]

                new_inputs = {}
                new_outputs = {}

                contains_history = False
                history_key_name = None
                for input_key, input_val in inputs.items():
                    if isinstance(input_val, History):
                        contains_history = True
                        assert history_key_name is None
                        history_key_name = input_key

                if contains_history:
                    s = "```json\n"
                    for i, message in enumerate(inputs[history_key_name].messages):
                        s += f"  {i}: {message}\n"
                    s += "```"
                    new_inputs["Context"] = s

                for input_key, input_val in inputs.items():
                    if contains_history and input_key == history_key_name:
                        continue

                    if isinstance(input_val, Type) and self.custom_instruction_proposer is not None:
                        # Keep original object - will be properly formatted when sent to reflection LM
                        new_inputs[input_key] = input_val
                    else:
                        new_inputs[input_key] = str(input_val)

                if isinstance(outputs, FailedPrediction):
                    s = "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                    s += "```\n"
                    s += outputs.completion_text + "\n"
                    s += "```\n\n"
                    new_outputs = s
                else:
                    for output_key, output_val in outputs.items():
                        new_outputs[output_key] = str(output_val)

                d = {"Inputs": new_inputs, "Generated Outputs": new_outputs}
                if isinstance(outputs, FailedPrediction):
                    adapter = ChatAdapter()
                    structure_instruction = ""
                    for dd in adapter.format(module.signature, [], {}):
                        structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                    d["Feedback"] = "Your output failed to parse. Follow this structure:\n" + structure_instruction
                    # d['score'] = self.failure_score
                else:
                    feedback_fn = self.feedback_map[pred_name]
                    fb = feedback_fn(
                        predictor_output=outputs,
                        predictor_inputs=inputs,
                        module_inputs=example,
                        module_outputs=prediction,
                        captured_trace=trace,
                    )
                    d["Feedback"] = fb["feedback"]
                    if fb["score"] != module_score:
                        if self.warn_on_score_mismatch:
                            logger.warning("The score returned by the metric with pred_name is different from the overall metric score. This can indicate 2 things: Either the metric is non-deterministic (e.g., LLM-as-judge, Semantic score, etc.) or the metric returned a score specific to pred_name that differs from the module level score. Currently, GEPA does not support predictor level scoring (support coming soon), and only requires a feedback text to be provided, which can be specific to the predictor or program level. GEPA will ignore the differing score returned, and instead use module level score. You can safely ignore this warning if using a semantic metric, however, if this mismatch is caused due to predictor scoring, please return module-level scores. To disable this warning, set warn_on_score_mismatch=False.")
                            self.warn_on_score_mismatch = False
                        fb["score"] = module_score

                items.append(d)

            if len(items) == 0:
                # raise Exception(f"No valid predictions found for module {module.signature}.")
                continue
            ret_d[pred_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d

    # TODO: The current DSPyAdapter implementation uses the GEPA default propose_new_texts.
    # We can potentially override this, to use the instruction proposal similar to MIPROv2.

    # def propose_new_texts(
    #     self,
    #     candidate: Dict[str, str],
    #     reflective_dataset: Dict[str, List[Dict[str, Any]]],
    #     components_to_update: List[str]
    # ) -> Dict[str, str]:
    #     if self.adapter.propose_new_texts is not None:
    #         return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

    #     from .instruction_proposal import InstructionProposalSignature
    #     new_texts: Dict[str, str] = {}
    #     for name in components_to_update:
    #         base_instruction = candidate[name]
    #         dataset_with_feedback = reflective_dataset[name]
    #         new_texts[name] = InstructionProposalSignature.run(
    #             lm=self.reflection_lm,
    #             input_dict={
    #                 "current_instruction_doc": base_instruction,
    #                 "dataset_with_feedback": dataset_with_feedback
    #             }
    #         )['new_instruction']
    #     return new_texts



import logging
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, TypedDict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
    callback_metadata: dict[str, Any] | None = None,
) -> list[TraceData]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    # Use `object.__getattribute__` to bypass the custom hook `Module.__getattribute__` so that we avoid
    # the warning that `forward` is not accessed through `__call__`.
    original_forward = object.__getattribute__(program, "forward")

    def patched_forward(program_to_use: Module, **kwargs):
        with dspy.context(trace=[]):
            try:
                return original_forward(**kwargs), dspy.settings.trace.copy()
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program_to_use.predictors():
                    if pred.signature == failed_signature:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                trace = dspy.settings.trace.copy()
                # Trace is Tuple[signature, inputs, prediction outputs]
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (present / expected),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                trace.append(
                    (
                        found_pred,
                        failed_inputs,
                        failed_pred,
                    )
                )

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following "
                        "the adapter's formatting."
                    )

                return failed_pred, trace

    program.forward = MethodType(patched_forward, program)

    try:
        results = evaluator(
            program,
            metric=wrapped_metric,
            callback_metadata=callback_metadata,
        ).results
    finally:
        program.forward = original_forward

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace = prediction
        except ValueError as ve:
            # TODO(GRPO Team): Often during GRPO bootstrapping, the LLM response does not follow dspy formatting. This
            # leads to a value error. To reproduce this issue, try Qwen/Qwen2.5-Coder-0.5B-Instruct with MATH dataset.
            # Proposal(Lakshya): We should capture the incorrectly-formatted LLM response, and store it in the trace,
            # and pass it to in the GRPO group with a high-negative user-configurable score.
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following "
                "dspy formatting."
            )
            if raise_on_error:
                raise ve
            else:
                continue
        data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data

````
