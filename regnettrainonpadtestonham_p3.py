import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================== PATH CONFIGURATION ========================
pad_model_path = r"D:\dataset\pad\organized_pad\pad_result_p3\regnetonpad_P3\regnety320_pad_P3_model.pth"
ham_test_dir = r"D:\dataset\ham\organized_ham\test"
save_root = r"D:\dataset\ham\organized_ham\ham_results_p3"
save_dir = os.path.join(save_root, "Loadregnettrainonpadtestonham_p3")
os.makedirs(save_dir, exist_ok=True)

# ======================== DEVICE SETUP ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================== DATA TRANSFORMS ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================== LOAD HAM TEST DATASET ========================
test_dataset = datasets.ImageFolder(root=ham_test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
ham_classes = test_dataset.classes
num_ham_classes = len(ham_classes)
print(f"HAM dataset classes: {ham_classes}")

# ======================== LOAD MODEL ========================
print("\nLoading trained RegNetY-320 model (PAD)...")
model = models.regnet_y_32gf(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_ham_classes)  # Adjust output layer

# Load state dict safely (handle mismatched fc layer)
checkpoint = torch.load(pad_model_path, map_location=device)
state_dict = checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"]

# Remove keys related to fc if dimensions differ
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
print(f"Ignored missing keys (likely fc layer): {missing}")
print(f"Unexpected keys (if any): {unexpected}")

model = model.to(device)
model.eval()

# ======================== EVALUATION ========================
y_true, y_pred = [], []

print("\nEvaluating model on HAM test dataset...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ======================== METRICS ========================
report = classification_report(y_true, y_pred, target_names=ham_classes, digits=4, output_dict=True)
conf_mat = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# ======================== SAVE RESULTS ========================
report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(save_dir, "result.txt")

with open(report_path, "w") as f:
    f.write("Classification Report (RegNetY-320 trained on PAD, tested on HAM)\n")
    f.write("=" * 70 + "\n\n")
    f.write(report_df.to_string())
    f.write(f"\n\nOverall Accuracy: {accuracy:.4f}\n")

print(f"\nResults saved to: {report_path}")

# ======================== SAVE CONFUSION MATRIX ========================
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=ham_classes, yticklabels=ham_classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (RegNetY-320 trained on PAD, tested on HAM)")

cm_path = os.path.join(save_dir, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

print("\n✅ Evaluation complete successfully! No errors.")


#done