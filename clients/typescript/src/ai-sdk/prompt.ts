import {
  generateText,
  streamText,
  type ModelMessage,
  type GenerateTextResult,
  type StreamTextResult,
  type ToolSet,
  type Output,
} from "ai";
import { requestContext } from "../gepa";
import type { Prompt as PromptInterface } from "../models";

type GenerateTextOptions = Parameters<typeof generateText>[0];
type StreamTextOptions = Parameters<typeof streamText>[0];

export class Prompt implements PromptInterface {
  systemPrompt: string;
  name: string = "";
  _isPrompt: boolean = true;

  constructor(systemPrompt: string) {
    this.systemPrompt = systemPrompt;
  }

  private prepareMessages(options: any): ModelMessage[] {
    if (options.messages) {
      if (options.messages.some((m: any) => m.role === "system")) {
        throw new Error(
          "System messages are not allowed in Prompt. Provide them via systemPrompt instead."
        );
      }
      return [
        { role: "system", content: this.systemPrompt } as ModelMessage,
        ...(options.messages as ModelMessage[]),
      ];
    }
    if (options.prompt) {
      return [
        { role: "system", content: this.systemPrompt } as ModelMessage,
        { role: "user", content: options.prompt } as ModelMessage,
      ];
    }
    throw new Error("Either prompt or messages must be provided.");
  }

  async generateText(
    options: GenerateTextOptions
  ): Promise<GenerateTextResult<any, any>> {
    const messages = this.prepareMessages(options);
    const store = requestContext.getStore();

    try {
      const result = await generateText({
        ...(options as any),
        system: undefined,
        prompt: undefined,
        messages,
      });

      if (store) {
        store.trace.push({
          predictor: this.name,
          input: messages,
          output: result.text,
        });
      }

      return result;
    } catch (e: any) {
      if (store) {
        // This catch block captures parsing errors (e.g. TypeValidationError, JSONParseError)
        // and connection errors, ensuring they are recorded in the GEPA trace.
        store.trace.push({
          predictor: this.name,
          input: messages,
          output: "",
          errorType: e.name || "Error",
          errorMessage: e.message || String(e),
          errorTraceback: e.stack || null,
        });
      }
      throw e;
    }
  }

  async streamText(
    options: StreamTextOptions
  ): Promise<StreamTextResult<any, any>> {
    const messages = this.prepareMessages(options);
    const store = requestContext.getStore();
    const predictorName = this.name;
    let traceRecorded = false; // Guard to ensure we only record to trace once per call

    const result = await streamText({
      ...(options as any),
      system: undefined,
      prompt: undefined,
      messages,
      onFinish: async (event: any) => {
        if (store && !traceRecorded) {
          traceRecorded = true;
          store.trace.push({
            predictor: predictorName,
            input: messages,
            output: event.text,
          });
        }
        if (options.onFinish) {
          await options.onFinish(event);
        }
      },
      onError: (error: any) => {
        if (store && !traceRecorded) {
          traceRecorded = true;
          store.trace.push({
            predictor: predictorName,
            input: messages,
            output: "",
            errorType: (error as any).name || "Error",
            errorMessage: (error as any).message || String(error),
            errorTraceback: (error as any).stack || null,
          });
        }
        if (options.onError) {
          options.onError(error);
        }
      },
    });

    return result;
  }
}
