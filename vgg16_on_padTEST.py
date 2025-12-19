# vgg16_on_pad_overlap.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Paths
# ===============================
MODEL_PATH = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\result_vgg16_new\vgg16_skin_cancer_model.pth"
PAD_TEST_DIR = r"D:\dataset\pad\organized_pad\test"
SAVE_DIR = r"D:\dataset\pad\organized_pad\vgg16_from_ISIC_overlap_result"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ===============================
# Transforms (same as ISIC training)
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# Load PAD dataset
# ===============================
pad_dataset = datasets.ImageFolder(PAD_TEST_DIR, transform=transform)
pad_loader  = torch.utils.data.DataLoader(pad_dataset, batch_size=32, shuffle=False)
pad_classes = pad_dataset.classes
print(f"📂 PAD classes ({len(pad_classes)}): {pad_classes}")

# ===============================
# ISIC-trained class names (9)
# ===============================
ISIC_CLASSES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "vascular lesion",
    "seborrheic keratosis",
    "squamous cell carcinoma",
]

# Build PAD -> ISIC mapping
pad_to_isic = {pad_classes.index(c): ISIC_CLASSES.index(c) for c in pad_classes if c in ISIC_CLASSES}
print("🔄 PAD → ISIC index mapping:", pad_to_isic)

# ===============================
# Load ISIC-trained VGG16 model
# ===============================
print("🧠 Loading ISIC-trained VGG16 model...")
model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, len(ISIC_CLASSES))  # 9 outputs
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# ===============================
# Inference (restricted to PAD classes)
# ===============================
print("🔍 Running inference on PAD test set...")
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in pad_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for t, p in zip(labels.numpy(), preds.cpu().numpy()):
            if t in pad_to_isic:  # only if PAD label exists in ISIC model
                all_labels.append(pad_to_isic[t])  # true label mapped to ISIC idx
                all_preds.append(p)

# ===============================
# Metrics
# ===============================
print("📊 Calculating metrics...")
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=[pad_to_isic[i] for i in range(len(pad_classes))])
report = classification_report(all_labels, all_preds,
                               labels=[pad_to_isic[i] for i in range(len(pad_classes))],
                               target_names=pad_classes, digits=4)

print(f"✅ Accuracy on PAD classes: {acc:.4f}")
print(report)

# ===============================
# Save results
# ===============================
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=pad_classes, yticklabels=pad_classes)
plt.title("Confusion Matrix (ISIC-trained VGG16 on PAD Classes)")
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

results_path = os.path.join(SAVE_DIR, "results.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("=== Evaluation of ISIC-trained VGG16 on PAD Classes ===\n\n")
    f.write(f"Accuracy on PAD 6 classes: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print("\n✅ Evaluation complete. Results saved at:")
print(f"   - Confusion Matrix: {cm_path}")
print(f"   - Results: {results_path}")
