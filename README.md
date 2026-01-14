# GEPA RPC

`gepa-rpc` is a standard interface for using GEPA (Genetic-Pareto prompt optimization) in any language or framework via remote calls to the GEPA engine. We also ship clients for specific frameworks to make integrarion easier. Currently, the only supported client is for the [Vercel AI SDK](https://sdk.vercel.ai/docs).

## Installation

Install in your ts/js project

```bash
npm install gepa-rpc
# or
bun add gepa-rpc
```

Install the cli.
First [install uv](https://docs.astral.sh/uv/getting-started/installation/)
Then install the gepa-rpc cli

```bash
uv tool install gepa-rpc
```

---

## Core Concepts

- `Prompt` is a wrapper for your AI client. (AI SDK in this case). It allows you to call `generateText`/ `streamText` how you normally would but it dynamically the system prompt for optimization.

- `Program` tracks all the `Prompt` components in your system and is the entry point for gepa prompt optimization.
- `Dataset` is a wrapper for your training data. It allows you to load your data from a JSONL file or an array of objects.
- `MetricFunction` is a function that you define that scores traces of your system's execution.
- `GEPA` is the optimizer. It uses the gepa-rpc cli to run the gepa engine and propose new prompts for optimization.

## Usage

### 1. Setup Your Dataset

Use the `Dataset` class to manage your training data. You can pass a path to a JSONL file or an array of records. The second argument specifies which fields from your dataset should be passed to your system's `forward` function.

```typescript
import { Dataset } from "gepa-rpc";

// Load from a JSONL file
const trainset = new Dataset("data/train.jsonl", ["question", "answer"]); // you can also pass in a dict mapping dataset keys to input keys

// Or use an array of objects
const trainset = new Dataset(
  [
    { ticket: "I can't log into my account.", label: "Login Issue" },
    { ticket: "Where is my order #123?", label: "Shipping" },
  ],
  ["ticket"] // These fields will be available in the 'forward' function
);
```

### 2. Define Your System (`Program`)

A `Program` tracks optimized system prompts for each AI component in your system. It automatically injects the correct system prompt for a component when you call
`program.<predictor_name>.generateText()`

#### Class-Based Approach

Shorthand approach with better type safety. (recommended)

```typescript
import { Program } from "gepa-rpc";
import { Prompt } from "gepa-rpc/ai-sdk";
import { openai } from "@ai-sdk/openai";
import { Output } from "ai";

class TicketClassifier extends Program<{ ticket: string }, string> {
  constructor() {
    super({
      classifier: new Prompt("Classify the support ticket into a category."),
    });
  }

  async forward(inputs: { ticket: string }): Promise<string> {
    const result = await (this.classifier as Prompt).generateText({
      model: openai("gpt-4o-mini"),
      prompt: `Ticket: ${inputs.ticket}`,
      output: Output.choice({
        options: ["Login Issue", "Shipping", "Billing", "General Inquiry"],
      }),
    });
    return result.output;
  }
}

const program = new TicketClassifier();
```

#### Functional Approach

Ideal for retrofitting an existing system. You can use a global Program object and replace `generateText`/ `streamText` calls with `program.predictor.generateText` / `program.predictor.streamText`.

```typescript
// program.ts
import { Program } from "gepa-rpc";
import { Prompt } from "gepa-rpc/ai-sdk";
import { openai } from "@ai-sdk/openai";
import { choose } from "./logic";

const program = new Program({
  judge: new Prompt(
    "Determine which response is better. Respond with A>B or B>A."
  ),
});

export default program;
```

```typescript
// logic.ts
import program from "./program";

const choose = async (
  question: string,
  response_A: string,
  response_B: string
) => {
  const result = await program.judge.generateText({
    model: openai("gpt-4o-mini"),
    prompt: `Question: ${question}\nA: ${response_A}\nB: ${response_B}`,
    output: Output.choice({
      options: ["A>B", "B>A"],
    }),
  });
  return result.output;
};
```

```typescript
// optimize.ts
import { GEPA } from "gepa-rpc";
import program from "./program";
import { choose } from "./logic";

program.setForward(
  async (inputs: {
    question: string;
    response_A: string;
    response_B: string;
  }) => {
    return await choose(inputs.question, inputs.response_A, inputs.response_B);
  }
);

// rest of optimization code..
```

### 3. Define Your Metric

The metric scores performance on a specific example. It can return a simple score or rich feedback to help the optimizer "reflect" on mistakes.

```typescript
import { type MetricFunction } from "gepa-rpc";

const metric: MetricFunction = (example, prediction) => {
  const isCorrect = example.label === prediction.output;
  return {
    score: isCorrect ? 1.0 : 0.0,
    feedback: isCorrect
      ? "Correctly labeled."
      : `Incorrectly labeled. Expected ${example.label} but got ${prediction.output}`,
  };
};
```

### 4. Run Optimization

Call `GEPA.compile` to start the optimization process. This spawns a reflective optimization loop where the system tries different prompt variations.

```typescript
// optimize.ts
import { GEPA } from "gepa-rpc";

const gepa = new GEPA({
  numThreads: 4, // Concurrent evaluation workers
  auto: "medium", // Optimization depth (light, medium, heavy)
  reflection_lm: "openai/gpt-4o", // Strong model used for reflection
});

const optimizedProgram = await gepa.compile(program, metric, trainset);

console.log(
  "Optimized Prompt:",
  (optimizedProgram.classifier as Prompt).systemPrompt
);
```

### 5. Persistence

You can save and load the state of your `Program` (the optimized prompts) to JSON files.

```typescript
// Save the optimized state
optimizedProgram.save("./optimized_prompts.json");

// Load it back later
const productionProgram = new TicketClassifier();
productionProgram.load("./optimized_prompts.json");
```

---

## Appendix

### Language Support

Currently, the only supported client is the [Vercel AI SDK](https://sdk.vercel.ai/docs). However, the `gepa-rpc` cli can be used with any language or framework. If you would like to have a client for your favorite framework/language, please open an issue or submit a pull request.

### Concurrency

Optimization uses a dynamic worker pool. If you set `numThreads` to 4, the TypeScript client will keep 4 LLM calls in flight simultaneously during evaluation, maximizing throughput.

### Development

If you are developing `gepa-rpc` locally, you can use the `GEPA_RPC_DEV=true` environment variable to run the CLI from the local source using `uv run` instead of `uvx`.

```bash
GEPA_RPC_DEV=true bun run your_optimization_script.ts
```

### Network Proocol

Currently, gepa-rpc uses HTTP request to comminicate between the cli and the framework client. We are working on a websocket based protocol for robustness.
