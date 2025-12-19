import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, RegNetForImageClassification
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Starting HuggingFace RegNet training for thesis...")

    # Paths (PAD dataset)
    print("📁 Setting paths...")
    base_dir = r"D:\dataset\pad\organized_pad"
    train_path = os.path.join(base_dir, "train")
    test_path = os.path.join(base_dir, "test")
    model_save_path = os.path.join(base_dir, "regnety_hf_model.pth")
    results_txt_path = os.path.join(base_dir, "result.txt")
    conf_matrix_path = os.path.join(base_dir, "confusion_matrix.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Hugging Face processor
    print("🌀 Loading HuggingFace AutoImageProcessor...")
    processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")

    # Load datasets with processor transform
    print("📂 Loading training and testing datasets...")
    train_dataset = ImageFolder(
        train_path,
        transform=lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
    )
    test_dataset = ImageFolder(
        test_path,
        transform=lambda img: processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
    )
    print(f"✅ Loaded {len(train_dataset)} train and {len(test_dataset)} test images.")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    print("🧠 Loading HuggingFace RegNet model...")
    model = RegNetForImageClassification.from_pretrained(
        "facebook/regnet-y-040",
        num_labels=len(train_dataset.classes),
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # Optimizer & Loss
    print("⚙️  Setting optimizer and loss...")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # History tracking
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    # Training loop
    print("🏋️  Starting training loop...")
    for epoch in range(10):
        print(f"\n🔁 Epoch {epoch+1}/10")
        model.train()
        running_loss, correct, total = 0, 0, 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 5 == 0:
                print(f"   Batch {i+1}: Loss = {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        # Validation/Test accuracy
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).logits
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        history["test_acc"].append(test_acc)

        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Final evaluation
    print("\n🔍 Final evaluation on test data...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).logits
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Accuracy & Metrics
    print("📊 Calculating metrics...")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)

    # Save model
    print("💾 Saving model...")
    torch.save(model.state_dict(), model_save_path)

    # Save confusion matrix
    print("🖼️  Saving confusion matrix...")
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
