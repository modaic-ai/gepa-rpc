export type Example = Record<string, any>;

interface Prediction<T> {
  output: T | null;
  errorType: string | null;
  errorMessage: string | null;
  errorTraceback: string | null;
}

interface TraceEntry<InputType, ResultType> {
  predictor: string;
  input: InputType;
  output: ResultType;
  errorType?: string | null;
  errorMessage?: string | null;
  errorTraceback?: string | null;
}

export interface ScoreWFeedback {
  score: number;
  feedback?: string;
}

export type MetricFunction = (
  gold: Example,
  prediction: Prediction<any>,
  trace?: TraceEntry<any, any>[],
  pred_name?: string,
  pred_trace?: TraceEntry<any, any>[]
) => ScoreWFeedback | number | Promise<ScoreWFeedback | number>;

type FeedbackFunction = (args: {
  predictor_output: any;
  predictor_inputs: any;
  module_inputs: any;
  module_outputs: any;
  captured_trace: TraceEntry<any, any>[];
}) => { feedback: string; score: number };

interface Trace<InputType, OutputType> {
  example_ind: number;
  example: Example;
  prediction: Prediction<OutputType>;
  trace: TraceEntry<any, any>[];
  score: ScoreWFeedback;
}

// Request Models //
interface EvaluateRequest {
  batch: Example[];
  candidate: Record<string, string>;
  capture_traces: boolean;
}

interface EvaluationBatch<TraceType, PredictionType> {
  outputs: PredictionType[];
  scores: number[];
  trajectories: TraceType[];
}

interface MakeReflectiveDatasetRequest {
  candidate: Record<string, string>;
  eval_batch: EvaluationBatch<Trace<any, any>, Prediction<any>>;
  components_to_update: string[];
}

interface ReflectiveExample {
  Inputs: Record<string, any>;
  "Generated Outputs": Record<string, any> | string;
  Feedback: string;
}

interface MakeReflectiveDatasetResponse {
  reflective_dataset: Record<string, ReflectiveExample[]>;
}

interface FinalizeRequest {
  best_candidate: Record<string, string>;
  results: any;
}

export interface Predict {
  name: string;
  systemPrompt: string;
}

export type {
  Prediction,
  TraceEntry,
  Trace,
  EvaluateRequest,
  EvaluationBatch,
  MakeReflectiveDatasetRequest,
  ReflectiveExample,
  MakeReflectiveDatasetResponse,
  FinalizeRequest,
  FeedbackFunction,
};
