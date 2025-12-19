# vgg16_on_ham_overlap.py
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
HAM_TEST_DIR = r"D:\dataset\ham\organized_ham\test"
SAVE_DIR = r"D:\dataset\ham\organized_ham\vgg16_from_ISIC_overlap_result"
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
# Load HAM dataset
# ===============================
ham_dataset = datasets.ImageFolder(HAM_TEST_DIR, transform=transform)
ham_loader  = torch.utils.data.DataLoader(ham_dataset, batch_size=32, shuffle=False)
ham_classes = ham_dataset.classes
print(f"📂 HAM classes ({len(ham_classes)}): {ham_classes}")

# ===============================
# Define overlap classes
# ===============================
OVERLAP_CLASSES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "vascular lesion",
]

# ISIC classes = 9 (HAM’s 7 + 2 extra)
ISIC_CLASSES = OVERLAP_CLASSES + ["seborrheic keratosis", "squamous cell carcinoma"]

# Build mapping HAM idx -> ISIC idx
ham_to_isic = {ham_classes.index(c): ISIC_CLASSES.index(c) for c in OVERLAP_CLASSES}
print("🔄 HAM → ISIC index mapping:", ham_to_isic)

# ===============================
# Load ISIC-trained VGG16 model
# ===============================
print("🧠 Loading ISIC-trained VGG16 model...")
num_isic_classes = len(ISIC_CLASSES)  # 9
model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, num_isic_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# ===============================
# Inference (only overlap classes)
# ===============================
print("🔍 Running inference on HAM test set...")
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in ham_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # keep only overlapping HAM classes
        for t, p in zip(labels.numpy(), preds.cpu().numpy()):
            if t in ham_to_isic:
                all_labels.append(ham_to_isic[t])  # map HAM true idx -> ISIC idx
                all_preds.append(p)

# ===============================
# Metrics
# ===============================
print("📊 Calculating metrics...")
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=range(len(OVERLAP_CLASSES)))
report = classification_report(all_labels, all_preds,
                               labels=range(len(OVERLAP_CLASSES)),
                               target_names=OVERLAP_CLASSES, digits=4)

print(f"✅ Accuracy on overlap classes: {acc:.4f}")
print(report)

# ===============================
# Save results
# ===============================
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=OVERLAP_CLASSES, yticklabels=OVERLAP_CLASSES)
plt.title("Confusion Matrix (ISIC-trained VGG16 on HAM Overlap Classes)")
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

results_path = os.path.join(SAVE_DIR, "results.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("=== Evaluation of ISIC-trained VGG16 on HAM (Overlap Classes) ===\n\n")
    f.write(f"Accuracy on overlap 7 classes: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print("\n✅ Evaluation complete. Results saved at:")
print(f"   - Confusion Matrix: {cm_path}")
print(f"   - Results: {results_path}")
