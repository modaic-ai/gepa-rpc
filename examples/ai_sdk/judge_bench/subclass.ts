// example ticket classifier. Using the subclass API.
import { GEPANode, type MetricFunction } from "gepa-rpc";
import { Predict } from "gepa-rpc/ai-sdk";
import { openai } from "@ai-sdk/openai";
import { Output } from "ai";

class MyAgent extends GEPANode {
  constructor() {
    super({
      judge: new Predict(
        "Read the question and determine which response is better. If A is better respond with A>B if B is better respond with B>A."
      ),
    });
  }

  async forward(inputs: {
    question: string;
    response_A: string;
    response_B: string;
  }): Promise<string> {
    const prompt = `Question: ${inputs.question}\n\nResponse A: ${inputs.response_A}\n\nResponse B: ${inputs.response_B}\n\n`;
    const result = await (this.judge as Predict).generateText({
      model: openai("gpt-4o-mini"),
      prompt: prompt,
      output: Output.choice({
        options: ["A>B", "B>A"],
      }),
    });
    return result.output;
  }
}

const node = new MyAgent();

const metric: MetricFunction = async (example, prediction) => {
  const expected = example.label === "A>B" ? "A" : "B";
  const predicted = prediction.output;
  const isCorrect = predicted === expected;
  return {
    score: isCorrect ? 1.0 : 0.0,
    feedback: isCorrect
      ? "Correct judgement."
      : `Incorrect. Predicted ${predicted} but expected ${expected} (label: ${example.label}).`,
  };
};
