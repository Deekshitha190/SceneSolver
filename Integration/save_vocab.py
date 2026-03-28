import json
from collections import Counter

# === Load your training summaries ===
with open(r"C:\Users\Praveen Kumar\Desktop\Scene_solver_app\Split_Dataset_final\Split_Dataset\train_summaries.json", "r") as f:
    summaries = [ann['summary'] for ann in json.load(f)]

# === Rebuild tokenizer ===
word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
counter = Counter()
for summary in summaries:
    counter.update(summary.lower().split())

for word, freq in counter.items():
    if freq >= 1:  # can adjust this
        word2idx[word] = len(word2idx)

# === Save the vocabulary ===
with open("tokenizer_vocab.json", "w") as f:
    json.dump(word2idx, f)

print("✅ tokenizer_vocab.json saved successfully.")
