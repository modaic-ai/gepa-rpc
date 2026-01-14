import { GEPA, GEPANode, Dataset, type MetricFunction } from "gepa-rpc";
import { Predict } from "gepa-rpc/ai-sdk";
import { openai } from "@ai-sdk/openai";
import { Output } from "ai";
console.log("GEPA_RPC_DEV env variable:", process.env.GEPA_RPC_DEV);

class TicketClassifier extends GEPANode<{ ticket: string }, string> {
  constructor() {
    super({
      classifier: new Predict("Classify the support ticket into a category."),
    });
  }

  async forward(inputs: { ticket: string }): Promise<string> {
    const result = await (this.classifier as Predict).generateText({
      model: openai("gpt-4o-mini"),
      prompt: `Ticket: ${inputs.ticket}`,
      output: Output.choice({
        options: [
          "Login Issue",
          "Shipping",
          "Refund/Returns",
          "Billing",
          "Technical Support",
          "Sales",
          "General Inquiry",
        ],
      }),
    });
    return result.output;
  }
}

const metric: MetricFunction = (example, prediction) => {
  const isCorrect = example.label === prediction.output;
  return {
    score: isCorrect ? 1.0 : 0.0,
    feedback: isCorrect
      ? "Correctly labeled."
      : `Incorrectly labeled. Expected ${example.label} but got ${prediction.output}`,
  };
};

const data = [
  {
    ticket:
      "I received a notification that my password was changed, but I didn't do it. Now I can't get in and I'm worried about my billing info being stolen.",
    label: "Login Issue",
  },
  {
    ticket:
      "The tracking for my replacement order #ABC says it's coming from overseas, but I was promised it would be here by tomorrow. I'd rather just cancel and get my money back if it's going to take weeks.",
    label: "Refund/Returns",
  },
  {
    ticket:
      "I'm trying to return this item but the return label link you emailed me gives a 404 error. Can you fix the website?",
    label: "Technical Support",
  },
  {
    ticket:
      "I saw an ad saying your new 'Pro' version has advanced API access. Does that include the Python SDK, or is that extra? I'm currently on the Free plan.",
    label: "Sales",
  },
  {
    ticket:
      "I was charged twice for my subscription this month. I think there might be a bug in your payment processing system that's causing duplicate transactions.",
    label: "Billing",
  },
  {
    ticket:
      "I'm interested in the enterprise plan for my team of 50. Who should I talk to about pricing?",
    label: "Sales",
  },
  {
    ticket:
      "My subscription renewed yesterday but I meant to cancel it. Can I get that charge reversed?",
    label: "Billing",
  },
  {
    ticket:
      "My product was damaged during shipping. I opened it and it was all wet and moldy. How can we proceed?",
    label: "Refund/Returns",
  },
  {
    ticket:
      "Is there a way to export my data as a CSV? I couldn't find the button in the settings.",
    label: "Technical Support",
  },
  {
    ticket:
      "My tracking number says delivered but I don't see anything on my porch.",
    label: "Shipping",
  },
];

const trainset = new Dataset(data, ["ticket"]);
const node = new TicketClassifier();

const gepa = new GEPA({
  numThreads: 2,
  auto: "light",
  reflection_lm: "openai/gpt-4o",
});

console.log("Starting optimization...");
const optimizedNode = await gepa.compile(node, metric, trainset);

optimizedNode.save("tickets_classifier.json");
console.log(
  "Optimized Prompt:",
  (optimizedNode.classifier as Predict).systemPrompt
);
