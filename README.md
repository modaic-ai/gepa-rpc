<p align="center">
  <img src="assets/logo.png" alt="GEPA RPC Logo" width="400">
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/gepa-rpc"><img src="https://img.shields.io/npm/v/gepa-rpc" alt="npm version"></a>
  <a href="https://pypi.org/project/gepa-rpc/"><img src="https://img.shields.io/pypi/v/gepa-rpc" alt="PyPI version"></a>
  <a href="https://github.com/modaic/gepa-rpc/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

<p align="center">
  <strong>Automatically optimize your AI prompts using genetic algorithms and Pareto optimization.</strong>
</p>

---

## Why GEPA?

Writing effective prompts is hard. Small wording changes can dramatically affect accuracy, but finding the right phrasing requires tedious trial and error.

**GEPA automates this.** You define a metric (e.g., "did it classify correctly?"), provide training examples, and GEPA evolves your prompts to maximize performance—no manual tuning required.

```
Before: "Classify the support ticket into a category."           → 72% accuracy
After:  "You are a support ticket routing system. Analyze the    → 94% accuracy
         customer's intent and classify into exactly one of
         the following categories..."
```

---

## Quick Start

Here's a complete example that optimizes a ticket classifier:

```typescript
import { Program, Dataset, GEPA, type MetricFunction } from "gepa-rpc";
import { Prompt } from "gepa-rpc/ai-sdk";
import { openai } from "@ai-sdk/openai";
import { Output } from "ai";

// 1. Define your AI system
class TicketClassifier extends Program<{ ticket: string }, string> {
  constructor() {
    super({
      classifier: new Prompt("Classify the support ticket into a category."),
    });
  }

  override async forward(inputs: { ticket: string }): Promise<string> {
    const result = await this.classifier.generateText({
      model: openai("gpt-4o-mini"),
      prompt: `Ticket: ${inputs.ticket}`,
      output: Output.choice({
        options: ["Login Issue", "Shipping", "Billing", "General Inquiry"],
      }),
    });
    return result.output as string;
  }
}

// 2. Load training data
const trainset = new Dataset(
  [
    { ticket: "I can't log into my account.", label: "Login Issue" },
    { ticket: "Where is my order #123?", label: "Shipping" },
    // ... more examples
  ],
  ["ticket"]
);

// 3. Define how to score predictions
const metric: MetricFunction = (example, prediction) => ({
  score: example.label === prediction.output ? 1.0 : 0.0,
  feedback:
    example.label === prediction.output
      ? "Correct!"
      : `Expected "${example.label}", got "${prediction.output}"`,
});

// 4. Run optimization
const gepa = new GEPA({ numThreads: 4, auto: "medium" });
const optimized = await gepa.compile(new TicketClassifier(), metric, trainset);

// 5. Use your optimized program
optimized.save("./optimized_prompts.json");
console.log("New prompt:", optimized.classifier.systemPrompt);
```

---

## Installation

GEPA has two components: a **TypeScript client** for your application and a **CLI** that runs the optimization engine.

### 1. Install the TypeScript client

```bash
npm install gepa-rpc
# or
bun add gepa-rpc
```

### 2. Install the CLI

First [install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv tool install gepa-rpc
```

---

## Core Concepts

| Concept            | Description                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| **Prompt**         | Wraps your AI calls (`generateText`/`streamText`). Injects the optimized system prompt automatically. |
| **Program**        | Container for all `Prompt` components in your system. Entry point for optimization.                   |
| **Dataset**        | Your training data—loaded from JSONL or passed as an array.                                           |
| **MetricFunction** | Scores each prediction. Returns a score (0-1) and optional feedback for the optimizer.                |
| **GEPA**           | The optimizer. Spawns the CLI and evolves prompts using Genetic-Pareto optimization.                  |

---

## Detailed Usage

### Loading Data

```typescript
import { Dataset } from "gepa-rpc";

// From a JSONL file
const trainset = new Dataset("data/train.jsonl", ["question", "answer"]);

// From an array
const trainset = new Dataset(
  [
    { ticket: "I can't log into my account.", label: "Login Issue" },
    { ticket: "Where is my order #123?", label: "Shipping" },
  ],
  ["ticket"]
); // Fields passed to forward()
```

### Defining Your Program

#### Class-Based (Recommended)

Best for new projects. Provides type safety and clean encapsulation.

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

  override async forward(inputs: { ticket: string }): Promise<string> {
    const result = await this.classifier.generateText({
      model: openai("gpt-4o-mini"),
      prompt: `Ticket: ${inputs.ticket}`,
      output: Output.choice({
        options: ["Login Issue", "Shipping", "Billing", "General Inquiry"],
      }),
    });
    return result.output as string;
  }
}

const program = new TicketClassifier();
```

#### Functional (For Existing Codebases)

Best for retrofitting GEPA into an existing system. Replace your `generateText`/`streamText` calls with `program.<name>.generateText`.

```typescript
// program.ts
import { Program } from "gepa-rpc";
import { Prompt } from "gepa-rpc/ai-sdk";

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
import { openai } from "@ai-sdk/openai";
import { Output } from "ai";

export const choose = async (
  question: string,
  response_A: string,
  response_B: string
) => {
  const result = await program.judge.generateText({
    model: openai("gpt-4o-mini"),
    prompt: `Question: ${question}\nA: ${response_A}\nB: ${response_B}`,
    output: Output.choice({ options: ["A>B", "B>A"] }),
  });
  return result.output;
};
```

```typescript
// optimize.ts
import { GEPA, Dataset } from "gepa-rpc";
import program from "./program";
import { choose } from "./logic";

program.setForward(async (inputs) => {
  return await choose(inputs.question, inputs.response_A, inputs.response_B);
});

const trainset = new Dataset("data/comparisons.jsonl", [
  "question",
  "response_A",
  "response_B",
]);
const metric = (example, prediction) => ({
  score: example.winner === prediction.output ? 1.0 : 0.0,
});

const gepa = new GEPA({ numThreads: 4, auto: "medium" });
await gepa.compile(program, metric, trainset);
```

### Writing Metrics

The metric function scores each prediction. Return a `score` (0-1) and optional `feedback` to help the optimizer understand mistakes.

```typescript
import { type MetricFunction } from "gepa-rpc";

const metric: MetricFunction = (example, prediction) => {
  const isCorrect = example.label === prediction.output;
  return {
    score: isCorrect ? 1.0 : 0.0,
    feedback: isCorrect
      ? "Correctly labeled."
      : `Incorrectly labeled. Expected "${example.label}" but got "${prediction.output}"`,
  };
};
```

### Running Optimization

```typescript
import { GEPA } from "gepa-rpc";

const gepa = new GEPA({
  numThreads: 4, // Concurrent evaluation workers
  auto: "medium", // Optimization depth: "light" | "medium" | "heavy"
  reflection_lm: "openai/gpt-4o", // Model used for reflection (optional)
});

const optimizedProgram = await gepa.compile(program, metric, trainset);
```

### Saving & Loading

```typescript
// Save optimized prompts
optimizedProgram.save("./optimized_prompts.json");

// Load in production
const productionProgram = new TicketClassifier();
productionProgram.load("./optimized_prompts.json");
```

---

## Appendix

### Language Support

Currently, the only supported client is the [Vercel AI SDK](https://sdk.vercel.ai/docs). The `gepa-rpc` CLI can work with any language or framework—contributions for other clients are welcome!

### Concurrency

Optimization uses a dynamic worker pool. Setting `numThreads: 4` keeps 4 LLM calls in flight simultaneously during evaluation, maximizing throughput.

### Local Development

To run the CLI from local source instead of the published package:

```bash
GEPA_RPC_DEV=true bun run your_optimization_script.ts
```

### Network Protocol

GEPA uses HTTP to communicate between the CLI and the TypeScript client. A WebSocket-based protocol for improved robustness is in development.
