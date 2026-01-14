import { GEPA, GEPANode, Dataset, type MetricFunction } from "gepa-rpc";
import { Predict } from "gepa-rpc/ai-sdk";
import { openai } from "@ai-sdk/openai";
import { Output } from "ai";

const trainset = new Dataset("examples/ai_sdk/judge_bench_train.jsonl", [
  "question",
  "response_A",
  "response_B",
  "label",
]);

const node = new GEPANode({
  judge: new Predict(
    "Read the question and determine which response is better. If A is better respond with A>B if B is better respond with B>A."
  ),
});

async function forward(inputs: {
  question: string;
  response_A: string;
  response_B: string;
}): Promise<string> {
  const prompt = `Question: ${inputs.question}\n\nResponse A: ${inputs.response_A}\n\nResponse B: ${inputs.response_B}\n\n`;
  const result = await (node.judge as Predict).generateText({
    model: openai("gpt-4o-mini"),
    prompt: prompt,
    output: Output.choice({
      options: ["A>B", "B>A"],
    }),
  });
  return result.output;
}

node.setForward(forward);

const metric: MetricFunction = (example, prediction) => {
  const isCorrect = example.label === prediction.output;

  return {
    score: isCorrect ? 1.0 : 0.0,
  };
};

const gepa = new GEPA({
  numThreads: 4,
  auto: "medium",
  reflection_lm: "openai/gpt-4o",
});
const optimizedNode = await gepa.compile(node, metric, trainset);

console.log("Optimized Prompt:", (optimizedNode.judge as Predict).systemPrompt);
