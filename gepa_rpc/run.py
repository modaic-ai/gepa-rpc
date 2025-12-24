import gepa
from datasets import load_dataset, Dataset
from .adapter import RPCAdapter
from typing import Any

dataset = load_dataset("mteb/banking77")
print(dataset.keys())

trainset = dataset["train"].shuffle(seed=43).select(range(100))
valset = dataset["test"].shuffle(seed=43).select(range(100))


class HFDataloader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def all_ids(self) -> list[int]:
        return list(range(len(self.dataset)))

    def fetch(self, ids: list[int]) -> list[dict[str, Any]]:
        return self.dataset.select(ids).to_list()

    def __len__(self) -> int:
        return len(self.dataset)


trainset = HFDataloader(trainset)
valset = HFDataloader(valset)

with open("tests/labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]


seed_prompt = {
    "system_prompt": f"Classify the following support ticket. Allowed categories: {', '.join(labels)}"
}

adapter = RPCAdapter("http://localhost:8000")
# Let's run GEPA optimization process.
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    max_metric_calls=150,  # <-- Set a budget
    reflection_lm="openai/gpt-5",  # <-- Use a strong model to reflect on mistakes and propose better prompts
)

print("GEPA Optimized Prompt:", gepa_result.best_candidate["system_prompt"])
