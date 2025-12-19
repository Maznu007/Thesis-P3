# skinlesnetonisic_P3.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# SkinLesNet that matches the provided architecture
class SkinLesNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesNet, self).__init__()
        # Feature extraction layers (Convolutional/Pooling blocks)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(p=0.5),
        )
        
        # Classifier layers (Flattening and Fully Connected)
        # Assuming input size of (224, 224) -> maxpool (4 times) -> 14x14 spatial size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    print("🚀 Starting SkinLesNet training on ISIC dataset...")

    # =================================================================
    # PATHS (UPDATED for ISIC and new output folder)
    # =================================================================
    base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    
    # Define input directories
    train_path = os.path.join(base_dir, "Train") # ISIC uses 'Train'
    test_path  = os.path.join(base_dir, "Test")  # ISIC uses 'Test'
    
    # Define output folder and create it
    OUTPUT_FOLDER = os.path.join(base_dir, "skinlesnetonisic_P3")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Define output file paths
    model_save_path   = os.path.join(OUTPUT_FOLDER, "skinlesnet_model.pth")
    results_txt_path  = os.path.join(OUTPUT_FOLDER, "result.txt")
    conf_matrix_path  = os.path.join(OUTPUT_FOLDER, "confusion_matrix.png")
    
    print(f"📂 Output will be saved to: {OUTPUT_FOLDER}")
    # =================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard Normalization (often used for custom models)
        transforms.Normalize([0.5]*3, [0.5]*3) 
    ])

    print("📂 Loading datasets...")
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset  = ImageFolder(test_path, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    num_classes   = len(train_dataset.classes)
    print(f"✅ Found {num_classes} classes: {train_dataset.classes}")

    model = SkinLesNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Lists to record metrics
    train_losses, train_accuracies, test_accuracies = [], [], []

    print("🏋️ Training started...")
    for epoch in range(20):  # 20 epochs
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

            running_loss += loss.item() * imgs.size(0) # Scale by batch size

            # Training accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1} Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        # Epoch metrics
        epoch_loss = running_loss / len(train_dataset)
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

        print(f"✅ Epoch {epoch+1}/20 | Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Final evaluation metrics
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
    plt.title("ISIC SkinLesNet Confusion Matrix")
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

        f.write("\nFinal Accuracy: {:.4f}\n\n".format(acc))
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nModel path: {model_save_path}\n")
        f.write(f"Confusion matrix: {conf_matrix_path}\n")
    print(f"📄 Results written to {results_txt_path}")

    print("✅ All tasks complete.")

if __name__ == "__main__":
    main()