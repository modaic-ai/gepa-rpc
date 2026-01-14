import * as fs from "node:fs";
import * as path from "node:path";

export class Dataset {
  data: string | Record<string, any>[];
  inputs: Record<string, string>;

  constructor(
    data: string | Record<string, any>[],
    inputs: Record<string, string> | string[]
  ) {
    this.data = data;
    if (Array.isArray(inputs)) {
      const inputsDict: Record<string, string> = {};
      for (const input of inputs as string[]) {
        inputsDict[input] = input;
      }
      this.inputs = inputsDict;
    } else {
      this.inputs = inputs;
    }
  }

  getInputs(example: Record<string, any>): Record<string, any> {
    const res: Record<string, any> = {};
    for (const [datasetKey, inputKey] of Object.entries(this.inputs)) {
      if (datasetKey in example) {
        res[inputKey] = example[datasetKey];
      }
    }
    return res;
  }

  toFile(runDir: string): string {
    if (typeof this.data === "string") {
      return this.data;
    }

    const datasetPath = path.join(runDir, "dataset.jsonl");
    const jsonl = this.data.map((item) => JSON.stringify(item)).join("\n");
    fs.writeFileSync(datasetPath, jsonl);
    return datasetPath;
  }
}
