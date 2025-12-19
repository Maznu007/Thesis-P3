import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ====================================================
# PATH CONFIGURATION
# ====================================================
isic_model_path = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\isic_p3_result\regnetonisicp3\regnety320_isic_P3_model.pth"
ham_test_path = r"D:\dataset\ham\organized_ham\test"
save_root = r"D:\dataset\ham\organized_ham\ham_results_p3"
save_folder = os.path.join(save_root, "Loadregnettrainonisictestonham_p3")

os.makedirs(save_folder, exist_ok=True)

# ====================================================
# DEVICE CONFIGURATION
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================
# DATA TRANSFORM
# ====================================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====================================================
# LOAD HAM TEST DATASET
# ====================================================
test_dataset = datasets.ImageFolder(ham_test_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
ham_classes = test_dataset.classes
num_ham_classes = len(ham_classes)
print(f"HAM dataset classes: {ham_classes}")

# ====================================================
# LOAD TRAINED REGNET MODEL (ISIC)
# ====================================================
print("\nLoading trained RegNetY-320 model (ISIC)...")

model = models.regnet_y_32gf(weights=None)
in_features = model.fc.in_features

# Safely load checkpoint
checkpoint = torch.load(isic_model_path, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# Remove final classifier weights to handle mismatch automatically
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

# Load backbone weights safely
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Backbone loaded successfully.")
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# Replace final classification layer for HAM dataset
model.fc = nn.Linear(in_features, num_ham_classes)
model.to(device)
model.eval()

print(f"✅ Model ready for evaluation — ISIC→HAM ({num_ham_classes} classes)\n")

# ====================================================
# EVALUATION ON HAM TEST DATASET
# ====================================================
y_true, y_pred = [], []

print("Evaluating on HAM test dataset...")

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ====================================================
# METRICS CALCULATION
# ====================================================
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=ham_classes, digits=4)
cm = confusion_matrix(y_true, y_pred)

# ====================================================
# SAVE RESULTS TO TEXT FILE
# ====================================================
result_txt_path = os.path.join(save_folder, "result.txt")
with open(result_txt_path, "w", encoding="utf-8") as f:
    f.write("=== Model Evaluation Report ===\n")
    f.write(f"Timestamp: {datetime.now()}\n\n")
    f.write(f"Model Path: {isic_model_path}\n")
    f.write(f"Test Dataset: {ham_test_path}\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

print(f"📄 Results saved to: {result_txt_path}")

# ====================================================
# SAVE CONFUSION MATRIX IMAGE
# ====================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ham_classes, yticklabels=ham_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - RegNetY320 (ISIC→HAM)')
cm_path = os.path.join(save_folder, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

print(f"🖼️ Confusion matrix saved to: {cm_path}")

print("\n✅ Evaluation complete!")
print(f"All results are saved under:\n{save_folder}")


#done