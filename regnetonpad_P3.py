import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import regnet_y_32gf
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Starting RegNetY training for thesis...")

    # Define the new base directory and results folder
    new_base_dir = r"D:\dataset\pad\organized_pad"
    results_folder = "hamonpad_P3"
    full_results_path = os.path.join(new_base_dir, results_folder)
    
    # Create the results folder if it doesn't exist
    if not os.path.exists(full_results_path):
        os.makedirs(full_results_path)
        print(f"📁 Created results folder: {full_results_path}")

    # Paths
    print("📁 Setting new paths...")
    # The 'Train' and 'Test' folders are directly under the 'new_base_dir'
    train_path = os.path.join(new_base_dir, "train")
    test_path = os.path.join(new_base_dir, "test")
    model_save_path = os.path.join(full_results_path, "regnety320_pad_P3_model.pth")
    results_txt_path = os.path.join(full_results_path, "regnety320_pad_P3_result.txt")
    conf_matrix_path = os.path.join(full_results_path, "regnety320_pad_P3_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Transforms
    print("🌀 Preparing image transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load datasets
    print("📂 Loading training and testing datasets...")
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    print(f"✅ Loaded {len(train_dataset)} train and {len(test_dataset)} test images.")
    print(f"Classes: {train_dataset.classes}")

    # DataLoaders
    print("📦 Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    print("🧠 Loading RegNetY-32GF model...")
    model = regnet_y_32gf(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    # Loss and Optimizer
    print("⚙️  Setting loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Track metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    # Training
    print("🏋️  Starting training loop...")
    epochs = 20
    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0
        correct, total = 0, 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy for training
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 5 == 0:
                print(f"   Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation/Test accuracy after each epoch
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        
        # Store the per-epoch metrics
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"✅ Epoch {epoch+1} completed | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Final Evaluation
    print("\n🔍 Final evaluation on test data...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Accuracy & Metrics
    print("📊 Calculating metrics...")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, zero_division=0)

    # Save model
    print("💾 Saving model...")
    torch.save(model.state_dict(), model_save_path)

    # Save confusion matrix
    print("🖼️  Saving confusion matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # Save results
    print("📄 Writing results to file...")
    with open(results_txt_path, "w") as f:
        f.write(f"Final Test Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")

        # Write per-epoch results
        f.write("Per-epoch results:\n")
        for i in range(len(history["train_loss"])):
            f.write(
                f"Epoch {i+1}: "
                f"Train Loss = {history['train_loss'][i]:.4f}, "
                f"Train Acc = {history['train_acc'][i]:.4f}, "
                f"Test Acc = {history['test_acc'][i]:.4f}\n"
            )

    print("\n✅ Training and evaluation complete. All results saved.")

if __name__ == "__main__":
    main()