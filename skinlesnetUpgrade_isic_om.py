import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
#  Upgraded SkinLesNet
# ===============================
class SkinLesNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 224 -> 112

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 112 -> 56

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 56 -> 28

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 28 -> 14

            nn.Dropout(p=0.4)
        )

        # Global Average Pooling instead of flattening huge tensor
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> 256 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ===============================
#  Main Training Script
# ===============================
def main():
    print("🚀 Starting SkinLesNet training on ISIC dataset...")

    # Paths (ISIC dataset)
    base_path = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    train_path = os.path.join(base_path, "Train")
    test_path  = os.path.join(base_path, "Test")
    model_save_path   = os.path.join(base_path, "skinlesnet_model.pth")
    results_txt_path  = os.path.join(base_path, "result.txt")
    conf_matrix_path  = os.path.join(base_path, "confusion_matrix.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Data Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset  = ImageFolder(test_path, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    num_classes   = len(train_dataset.classes)
    print(f"✅ Found {num_classes} classes: {train_dataset.classes}")

    # Initialize model
    model = SkinLesNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Tracking metrics
    train_losses, train_accuracies, test_accuracies = [], [], []

    # Training loop
    print("🏋️ Training started...")
    for epoch in range(25):  # more epochs
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Training accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1} Batch {i+1}: Loss = {loss.item():.4f}")

        # Epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Evaluate on test set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        test_acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(test_acc)

        print(f"✅ Epoch {epoch+1}/25 | Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | Test Acc: {test_acc:.4f}")

        # Step scheduler
        scheduler.step()

    # Final evaluation
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    acc = accuracy_score(all_labels, all_preds)

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")

    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"🖼️ Confusion matrix saved to {conf_matrix_path}")

    # Save results
    with open(results_txt_path, "w") as f:
        for e in range(len(train_losses)):
            f.write(f"Epoch {e+1}: "
                    f"Train Loss={train_losses[e]:.4f}, "
                    f"Train Acc={train_accuracies[e]:.4f}, "
                    f"Test Acc={test_accuracies[e]:.4f}\n")

        f.write("\nFinal Test Accuracy: {:.4f}\n\n".format(acc))
        f.write("Classification Report:\n")
        f.write(report)
    print(f"📄 Results written to {results_txt_path}")

    print("✅ All tasks complete.")

if __name__ == "__main__":
    main()
