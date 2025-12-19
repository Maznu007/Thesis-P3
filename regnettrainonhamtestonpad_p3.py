import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime

# ===============================
# Paths
# ===============================
ham_model_path = r"D:\dataset\ham\organized_ham\ham_regnetBase_result\regnety320_model.pth"
pad_test_path = r"D:\dataset\pad\organized_pad\test"
save_root = r"D:\dataset\pad\organized_pad\pad_result_p3"
save_folder = os.path.join(save_root, "Loadregnettrainonhamtestonpad_p3")

os.makedirs(save_folder, exist_ok=True)

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Data Transform
# ===============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===============================
# Load PAD Test Dataset
# ===============================
test_dataset = datasets.ImageFolder(pad_test_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
pad_classes = test_dataset.classes
num_pad_classes = len(pad_classes)
print(f"PAD dataset classes: {pad_classes}")

# ===============================
# Load Trained RegNet Model
# ===============================
print("Loading trained RegNetY-320 model...")
model = models.regnet_y_32gf(weights=None)
in_features = model.fc.in_features

# Load checkpoint
checkpoint = torch.load(ham_model_path, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# Remove classifier weights (fc layer) to avoid mismatch
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

# Load backbone weights safely
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Replace final layer for PAD dataset (handle class mismatch)
model.fc = nn.Linear(in_features, num_pad_classes)

model.to(device)
model.eval()

print(f"✅ Model loaded successfully with backbone from HAM and new head for PAD ({num_pad_classes} classes).")

# ===============================
# Testing
# ===============================
y_true, y_pred = [], []

print("Evaluating on PAD test dataset...")

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ===============================
# Metrics
# ===============================
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=pad_classes, digits=4)
cm = confusion_matrix(y_true, y_pred)

# ===============================
# Save Results
# ===============================
result_txt_path = os.path.join(save_folder, "result.txt")

with open(result_txt_path, "w", encoding="utf-8") as f:
    f.write("=== Model Evaluation Report ===\n")
    f.write(f"Timestamp: {datetime.now()}\n\n")
    f.write(f"Model Path: {ham_model_path}\n")
    f.write(f"Test Dataset: {pad_test_path}\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

print(f"Results saved to: {result_txt_path}")

# ===============================
# Confusion Matrix
# ===============================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=pad_classes, yticklabels=pad_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - RegNetY320 (HAM→PAD)')
cm_path = os.path.join(save_folder, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

print(f"Confusion matrix saved to: {cm_path}")
print("✅ Evaluation complete!")
print(f"All results saved under:\n{save_folder}")


#done