# pad_vgg16_final.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# 0. Paths (PAD dataset)
# ===============================
BASE_DIR  = r"D:\dataset\pad\organized_pad"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

print("[1/10] Verifying dataset paths...")
assert os.path.exists(TRAIN_DIR), f"Train path not found: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR),  f"Test path not found:  {TEST_DIR}"
print("       Paths OK.")

# ===============================
# 1. Device setup
# ===============================
print("[2/10] Setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"       Using device: {device}")

# ===============================
# 2. Data transforms
# ===============================
print("[3/10] Building transforms...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),                 # VGG16 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

# ===============================
# 3. Datasets & Loaders
# ===============================
print("[4/10] Loading datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"       Classes ({num_classes}): {class_names}")

# ===============================
# 4. Load VGG16 (random init)
# ===============================
print("[5/10] Building VGG16 (weights=None)...")
model = models.vgg16(weights=None)          # random initialization
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)
print("       VGG16 ready.")

# ===============================
# 5. Loss & Optimizer
# ===============================
print("[6/10] Setting up loss & optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# Helper: evaluate one pass
# ===============================
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    avg_acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, avg_acc

# ===============================
# 6. Training
# ===============================
print("[7/10] Starting training loop...")
num_epochs = 20
epoch_logs = []  # will write to result.txt

for epoch in range(num_epochs):
    print(f"       Epoch {epoch+1}/{num_epochs} - training...")
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0 or batch_idx == len(train_loader):
            print(f"         Batch {batch_idx}/{len(train_loader)}")

    # Per-epoch Train metrics
    train_loss = running_loss / len(train_dataset)
    train_acc  = 100.0 * correct / total if total > 0 else 0.0

    # Per-epoch Test metrics
    print("       Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    log_line = (f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print("       " + log_line)
    epoch_logs.append(log_line)

print("[8/10] Training finished.")

# ===============================
# 7. Save model
# ===============================
model_path = os.path.join(BASE_DIR, "vgg16_skin_cancer_model.pth")
torch.save(model.state_dict(), model_path)
print(f"[9/10] Model saved at: {model_path}")

# ===============================
# 8. Final Evaluation (preds for report & CM)
# ===============================
print("[10/10] Generating classification report and confusion matrix...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"       Confusion matrix saved at: {cm_path}")

# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names)
results_path = os.path.join(BASE_DIR, "result.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("Classification Report (Test Set):\n")
    f.write(report + "\n\n")
    f.write("Per-epoch metrics:\n")
    for line in epoch_logs:
        f.write(line + "\n")
    f.write(f"\nModel path: {model_path}\n")
    f.write(f"Confusion matrix: {cm_path}\n")

print(f"       Results saved at: {results_path}")
print("All done.")
