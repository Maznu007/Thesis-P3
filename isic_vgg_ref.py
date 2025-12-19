# isic_vgg16_transfer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 0. Paths
# ===============================
BASE_DIR  = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR  = os.path.join(BASE_DIR, "Test")

assert os.path.exists(TRAIN_DIR), f"Train path not found: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR),  f"Test path not found:  {TEST_DIR}"

# ===============================
# 1. Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# 2. Data transforms (+ augmentation)
# ===============================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# ===============================
# 3. Load pretrained VGG16 + modify
# ===============================
print("Loading pretrained VGG16...")
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Freeze early convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, num_classes)

model = model.to(device)

# ===============================
# 4. Loss, Optimizer, Scheduler
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ===============================
# Helper: evaluate
# ===============================
def evaluate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(loader.dataset), 100.0 * correct / total

# ===============================
# 5. Training loop
# ===============================
num_epochs = 30
epoch_logs = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_dataset)
    train_acc  = 100.0 * correct / total

    test_loss, test_acc = evaluate(model, test_loader)
    scheduler.step()

    log_line = (f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(log_line)
    epoch_logs.append(log_line)

print("Training finished.")

# ===============================
# 6. Save model
# ===============================
model_path = os.path.join(BASE_DIR, "vgg16_transfer_finetuned.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved at: {model_path}")

# ===============================
# 7. Final Evaluation (report + confusion matrix)
# ===============================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved at: {cm_path}")

# Classification report
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

print(f"Results saved at: {results_path}")
