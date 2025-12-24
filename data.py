from datasets import load_dataset

dataset = load_dataset("mteb/banking77", split="train")

ds100 = dataset.shuffle(seed=43).select(range(100))

print(ds100[0])
print(type(ds100[0]))

classes = ds100.unique("label_text")
print(classes)
