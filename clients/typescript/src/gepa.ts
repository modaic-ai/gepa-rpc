import { AsyncLocalStorage } from "node:async_hooks";
import { nanoid } from "nanoid";
import * as fs from "node:fs";
import * as path from "node:path";
import { spawn } from "node:child_process";
import { stringify } from "yaml";
import { createServer } from "./server";
import type { Example } from "./models";
import { Dataset } from "./dataset";
import type {
  Prediction,
  ScoreWFeedback,
  TraceEntry,
  Trace,
  EvaluationBatch,
  ReflectiveExample,
  FeedbackFunction,
  MetricFunction,
  Prompt,
} from "./models";

export const requestContext = new AsyncLocalStorage<{
  runId: string;
  trace: TraceEntry<any, any>[];
}>();

interface RunResult {
  prediction: Prediction<any>;
  trace: TraceEntry<any, any>[];
}

export interface GEPAOptions {
  // Budget configuration
  auto?: "light" | "medium" | "heavy";
  max_full_evals?: number;
  max_metric_calls?: number;

  // Reflection configuration
  reflection_minibatch_size?: number;
  candidate_selection_strategy?: "pareto" | "current_best";
  reflection_lm?: string;
  skip_perfect_score?: boolean;
  add_format_failure_as_feedback?: boolean;
  component_selector?: "round_robin" | "all";

  // Merge-based configuration
  use_merge?: boolean;
  max_merge_invocations?: number;

  // Evaluation configuration
  numThreads?: number; // Renamed from numWorkers
  failure_score?: number;
  perfect_score?: number;

  // Logging & Tracking
  log_dir?: string;
  track_stats?: boolean;
  use_wandb?: boolean;
  wandb_api_key?: string;
  wandb_init_kwargs?: Record<string, any>;
  track_best_outputs?: boolean;
  use_mlflow?: boolean;

  // Reproducibility
  seed?: number;

  // Additional configuration
  feedback_map?: Record<string, FeedbackFunction>;
}

export class GEPA {
  options: GEPAOptions;
  metric?: MetricFunction;
  dataset?: Dataset;
  private finalizeResolve?: (result: any) => void;

  constructor(options: GEPAOptions) {
    this.options = {
      numThreads: 4, // Default value
      perfect_score: 1.0,
      skip_perfect_score: true,
      use_merge: true,
      max_merge_invocations: 5,
      candidate_selection_strategy: "pareto",
      reflection_minibatch_size: 3,
      ...options,
    };
  }

  private calculateAutoBudget(
    numPreds: number,
    numCandidates: number,
    valsetSize: number
  ): number {
    const minibatchSize = 35;
    const fullEvalSteps = 5;

    const numTrials = Math.floor(
      Math.max(
        2 * (numPreds * 2) * Math.log2(numCandidates),
        1.5 * numCandidates
      )
    );

    if (numTrials < 0 || valsetSize < 0) {
      throw new Error("num_trials and valset_size must be >= 0.");
    }

    const V = valsetSize;
    const N = numTrials;
    const M = minibatchSize;
    const m = fullEvalSteps;

    // Initial full evaluation on the default program
    let total = V;

    // Assume upto 5 trials for bootstrapping each candidate
    total += numCandidates * 5;

    // N minibatch evaluations
    total += N * M;
    if (N === 0) {
      return total;
    }

    // Periodic full evals occur when trial_num % (m+1) == 0, where trial_num runs 2..N+1
    const periodicFulls = Math.floor((N + 1) / m) + 1;
    // If 1 <= N < m, the code triggers one final full eval at the end
    const extraFinal = N < m ? 1 : 0;

    total += (periodicFulls + extraFinal) * V;
    return total;
  }

  async finalize(bestCandidate: Record<string, string>, results: any) {
    if (this.finalizeResolve) {
      this.finalizeResolve({ bestCandidate, results });
    }
  }

  async compile(
    program: Program,
    metric: MetricFunction,
    dataset: Dataset,
    valset?: Dataset
  ): Promise<Program> {
    this.metric = metric;
    this.dataset = dataset;
    const server = createServer(this, program);
    const port = 8000;

    await server.listen({ port, host: "0.0.0.0" });
    console.log(`GEPA optimization server started on port ${port}`);

    // Handle budgeting
    const AUTO_RUN_SETTINGS = {
      light: { n: 6 },
      medium: { n: 12 },
      heavy: { n: 18 },
    };

    if (this.options.auto) {
      const numPreds = Object.keys(program._predictors).length;
      const numCandidates = AUTO_RUN_SETTINGS[this.options.auto].n;
      const valsetSize = valset ? valset.data.length : dataset.data.length;
      this.options.max_metric_calls = this.calculateAutoBudget(
        numPreds,
        numCandidates,
        valsetSize
      );
    } else if (this.options.max_full_evals) {
      const trainSize = dataset.data.length;
      const valSize = valset ? valset.data.length : dataset.data.length;
      this.options.max_metric_calls =
        this.options.max_full_evals * (trainSize + valSize);
    }

    // Create a temporary directory for optimization artifacts
    const logDir =
      this.options.log_dir || path.join(process.cwd(), ".gepa_run");
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    try {
      // 1. Prepare dataset path
      const datasetPath = dataset.toFile(logDir);
      const valsetPath = valset ? valset.toFile(logDir) : datasetPath;

      // 2. Prepare seed candidate from program predictors
      const seedCandidate: Record<string, string> = {};
      for (const [name, predictor] of Object.entries(program._predictors)) {
        seedCandidate[name] = predictor.systemPrompt;
      }

      // 3. Generate config.yaml
      const configPath = path.join(logDir, "config.yaml");
      const { feedback_map, numThreads, ...restOptions } = this.options;
      const config: Record<string, any> = {
        ...restOptions,
        trainset: datasetPath,
        valset: valsetPath,
        seed_candidate: seedCandidate,
        adapter: {
          base_url: `http://localhost:${port}`,
        },
      };

      const yamlContent = stringify(config);
      fs.writeFileSync(configPath, yamlContent);

      // 3. Spawn CLI
      console.log("Starting optimization process...");

      const optimizationPromise = new Promise<{
        bestCandidate: Record<string, string>;
        results: any;
      }>((resolve) => {
        this.finalizeResolve = resolve;
      });

      const isDev = process.env.GEPA_RPC_DEV === "true";
      const cmd = isDev ? "uv" : "uvx";
      const args = isDev
        ? ["run", "gepa-rpc", "--port", String(port), "--config", configPath]
        : ["gepa-rpc", "--port", String(port), "--config", configPath];

      const pythonProcess = spawn(cmd, args, {
        stdio: "inherit",
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      return await new Promise((resolve, reject) => {
        pythonProcess.on("error", (err) => {
          console.error("Failed to start optimization process:", err);
          reject(err);
        });

        optimizationPromise.then(({ bestCandidate, results }) => {
          // Update program with best candidate
          for (const [name, prompt] of Object.entries(bestCandidate)) {
            if (program._predictors[name]) {
              program._predictors[name].systemPrompt = prompt;
            }
          }
          console.log("Optimization completed via /finalize callback.");
          resolve(program);
        });

        pythonProcess.on("close", (code) => {
          if (code !== 0) {
            reject(new Error(`Optimization process failed with code ${code}`));
          }
        });
      });
    } catch (error) {
      console.error("Optimization failed:", error);
      throw error;
    } finally {
      await server.close();
      console.log("GEPA optimization server stopped");
    }
  }

  async make_reflective_dataset(
    candidate: Record<string, string>,
    eval_batch: EvaluationBatch<Trace<any, any>, Prediction<any>>,
    components_to_update: string[]
  ): Promise<Record<string, ReflectiveExample[]>> {
    const reflectiveDataset: Record<string, ReflectiveExample[]> = {};

    for (const componentName of components_to_update) {
      reflectiveDataset[componentName] = [];

      for (const trajectory of eval_batch.trajectories) {
        // Find trace entries where this component was used
        const relevantEntries = trajectory.trace.filter(
          (t) => t.predictor === componentName
        );

        for (const entry of relevantEntries) {
          let feedback = "";

          if (entry.errorMessage) {
            feedback = `Error occurred during execution: ${entry.errorType}: ${entry.errorMessage}`;
            if (entry.errorTraceback) {
              feedback += `\nTraceback: ${entry.errorTraceback}`;
            }
          } else {
            if (this.metric) {
              // Create sub-trace for the predictor
              const pred_trace = [entry];
              const metricVal = await this.metric(
                trajectory.example,
                trajectory.prediction,
                trajectory.trace,
                componentName,
                pred_trace
              );

              if (typeof metricVal === "number") {
                feedback = `This trajectory got a score of ${metricVal}.`;
              } else {
                feedback =
                  metricVal.feedback ??
                  `This trajectory got a score of ${metricVal.score}.`;
              }
            } else {
              const feedbackFn = this.options.feedback_map?.[componentName];
              if (feedbackFn) {
                const res = feedbackFn({
                  predictor_output: entry.output,
                  predictor_inputs: entry.input,
                  module_inputs: trajectory.example,
                  module_outputs: trajectory.prediction.output,
                  captured_trace: trajectory.trace,
                });
                feedback = res.feedback;
              } else {
                feedback = `This trajectory got a score of ${trajectory.score.score}.`;
              }
            }
          }

          reflectiveDataset[componentName].push({
            Inputs: { input: entry.input },
            "Generated Outputs": {
              output: entry.output,
            },
            Feedback: feedback,
          });
        }
      }
    }

    return reflectiveDataset;
  }
}

export class Program<Input = any, Output = any> {
  [key: string]: any;
  private _forward?: (input: Input) => Promise<Output>;
  _predictors: Record<string, Prompt>;

  constructor(
    predictors: Record<string, Prompt>,
    forward?: (input: Input) => Promise<Output>
  ) {
    this._predictors = predictors;
    for (const [name, predictor] of Object.entries(predictors)) {
      predictor.name = name;
      this[name] = predictor;
    }
    this._forward = forward;
  }

  clone(): Program<Input, Output> {
    const clonedPredictors = JSON.parse(JSON.stringify(this._predictors));
    return new Program(clonedPredictors, this.forward);
  }

  save(filePath: string): void {
    const state: Record<string, string> = {};
    for (const [name, predictor] of Object.entries(this._predictors)) {
      state[name] = predictor.systemPrompt;
    }
    fs.writeFileSync(filePath, JSON.stringify(state, null, 2));
  }

  load(filePath: string): void {
    const content = fs.readFileSync(filePath, "utf-8");
    const state = JSON.parse(content);
    for (const [name, prompt] of Object.entries(state)) {
      if (this._predictors[name]) {
        this._predictors[name].systemPrompt = prompt as string;
      }
    }
  }

  setForward(forward: (inputs: Input) => Promise<Output>) {
    this._forward = forward;
  }

  async forward(inputs: Input): Promise<Output> {
    if (!this._forward) {
      throw new Error("Forward function is not set");
    }
    return await this._forward(inputs);
  }

  async run(inputs: Input): Promise<RunResult> {
    const trace: TraceEntry<any, any>[] = [];
    let prediction: Prediction<any>;

    try {
      const output = await requestContext.run(
        { runId: nanoid(), trace },
        async () => {
          return await this.forward!(inputs);
        }
      );
      prediction = {
        output,
        errorType: null,
        errorMessage: null,
        errorTraceback: null,
      };
    } catch (e: any) {
      prediction = {
        output: null,
        errorType: e.name || "Error",
        errorMessage: e.message || String(e),
        errorTraceback: e.stack || null,
      };
    }

    return {
      prediction,
      trace,
    };
  }

  async evaluate(
    batch: Example[],
    candidate: Record<string, string>,
    capture_traces: boolean,
    numWorkers: number,
    metric: MetricFunction,
    getInputs?: (ex: Record<string, any>) => Record<string, any>
  ): Promise<EvaluationBatch<Trace<any, any>, Prediction<any>>> {
    // 1. Update candidate prompts
    for (const [name, prompt] of Object.entries(candidate)) {
      if (this[name] && typeof this[name] !== "function") {
        (this[name] as any).systemPrompt = prompt;
      }
    }

    // 2. Run batch with worker pool
    const outputs: Prediction<any>[] = new Array(batch.length);
    const scores: number[] = new Array(batch.length);
    const trajectories: Trace<any, any>[] = new Array(batch.length);

    let currentIndex = 0;
    const workers = Array.from({ length: numWorkers }).map(async () => {
      while (currentIndex < batch.length) {
        const index = currentIndex++;
        if (index >= batch.length) break;

        const example = batch[index]!;
        const inputs = getInputs ? getInputs(example) : example;
        const runResult = await this.run(inputs as Input);

        let metricResult: ScoreWFeedback;
        try {
          const metricVal = await metric(
            example,
            runResult.prediction,
            runResult.trace
          );
          if (typeof metricVal === "number") {
            metricResult = {
              score: metricVal,
              feedback: `This trajectory got a score of ${metricVal}.`,
            };
          } else {
            metricResult = {
              score: metricVal.score,
              feedback:
                metricVal.feedback ??
                `This trajectory got a score of ${metricVal.score}.`,
            };
          }
        } catch (e) {
          console.error(`Metric evaluation failed for example ${index}:`, e);
          metricResult = { score: 0, feedback: "Metric evaluation failed." };
        }

        outputs[index] = runResult.prediction;
        scores[index] = metricResult.score;

        if (capture_traces) {
          trajectories[index] = {
            example_ind: index,
            example: example,
            prediction: runResult.prediction,
            trace: runResult.trace,
            score: {
              score: metricResult.score,
              feedback: metricResult.feedback,
            },
          };
        }
      }
    });

    await Promise.all(workers);

    return {
      outputs,
      scores,
      trajectories: capture_traces ? trajectories : ([] as Trace<any, any>[]),
    };
  }
}
