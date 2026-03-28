import matplotlib.pyplot as plt

epochs_range = range(1, len(sum_train_losses) + 1)

# === Plot 1: Loss (Train vs Val) ===
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, sum_train_losses, label='Train Loss', color='blue', marker='o')
plt.plot(epochs_range, sum_val_losses, label='Val Loss', color='orange', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("T5 Decoder - Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("T5_Loss_Curve.png")
plt.show()

# === Plot 2: Accuracy (Train vs Val) ===
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, [a * 100 for a in sum_train_accuracies], label='Train Accuracy', color='green', marker='o')
plt.plot(epochs_range, [a * 100 for a in sum_val_accuracies], label='Val Accuracy', color='red', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("T5 Decoder - Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("T5_Accuracy_Curve.png")
plt.show()

# === Plot 3: BLEU Score ===
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, bleu_scores, label='BLEU Score', color='purple', marker='o')
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.title("T5 Decoder - BLEU Score per Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("T5_BLEU_Score.png")
plt.show()

# === Plot 4: ROUGE-L Score ===
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, rouge_scores, label='ROUGE-L Score', color='brown', marker='o')
plt.xlabel("Epoch")
plt.ylabel("ROUGE-L Score")
plt.title("T5 Decoder - ROUGE-L Score per Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("T5_ROUGE_Score.png")
plt.show()