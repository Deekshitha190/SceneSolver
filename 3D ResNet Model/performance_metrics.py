from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ====================== CLASSIFICATION METRICS (3D ResNet) ======================
print("\n=== Final Evaluation on Validation Set (3D ResNet Classification) ===")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for clips, labels in val_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        pred = torch.argmax(outputs, dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Class names
class_names = list(class_to_idx.keys())

# --- Classification Report ---
report_str = classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4,
    zero_division=0
)
report_dict = classification_report(
    all_labels, all_preds,
    target_names=class_names,
    digits=4,
    output_dict=True,
    zero_division=0
)
print(report_str)

# --- Save Classification Report as Image ---
def save_classification_report_as_image(report_str, filename="Classification_Report_(3D_ResNet).png"):
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.text(0.01, 0.99, report_str, {'fontsize': 12, 'color': 'darkslategray'},
             fontproperties='monospace', verticalalignment='top')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_classification_report_as_image(report_str)

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("Confusion_Matrix_(3D_ResNet).png", dpi=300)
plt.close()

# --- Bar Plots for Precision, Recall, F1 Score ---
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    values = [report_dict[cls][metric] for cls in class_names]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(class_names, values, color='cadetblue')
    plt.ylim(0, 1)
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} per Class")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"{metric.capitalize()}_Bar_Plot_(3D_ResNet).png", dpi=300)
    plt.close()

# ====================== LOSS & ACCURACY CURVES ======================
epoch_range = list(range(1, len(train_losses) + 1))
epoch_ticks = list(range(1, len(train_losses) + 1, max(1, len(train_losses) // 5)))

# --- Classification Loss Curve ---
plt.figure(figsize=(6, 5))
plt.plot(epoch_range, train_losses, label='Train Loss', color='darkorange')
plt.plot(epoch_range, val_losses, label='Val Loss', color='royalblue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve (3D ResNet)')
plt.xticks(epoch_ticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Loss_vs_Epochs_(3D_ResNet).png", dpi=300)
plt.close()

# --- Classification Accuracy Curve ---
plt.figure(figsize=(6, 5))
plt.plot(epoch_range, [acc * 100 for acc in train_accuracies], label='Train Acc', color='darkgreen')
plt.plot(epoch_range, [acc * 100 for acc in val_accuracies], label='Val Acc', color='crimson')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve (3D ResNet)')
plt.xticks(epoch_ticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Accuracy_vs_Epochs_(3D_ResNet).png", dpi=300)
plt.close()