import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import regnet_y_32gf
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Starting ISIC evaluation with HAM-trained RegNetY-32GF...")

    # Paths
    print("📁 Setting paths...")
    base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    test_path = os.path.join(base_dir, "Test")  # Your ISIC test folder (7 classes only)
    model_ckpt_path = r"D:\dataset\ham\organized_ham\ham_regnetBase_result\regnety320_model.pth"
    save_dir = os.path.join(base_dir, "regnety320_on_ISIC_result")
    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, "regnety320_on_ISIC.pth")
    results_txt_path = os.path.join(save_dir, "results_ISIC.txt")
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix_ISIC.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Transforms (must match training time)
    print("🌀 Preparing image transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load ISIC test dataset
    print("📂 Loading ISIC test dataset...")
    test_dataset = ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"✅ Loaded {len(test_dataset)} test images across {len(test_dataset.classes)} classes.")
    print("   Classes:", test_dataset.classes)

    # Load model
    print("🧠 Loading HAM-trained RegNetY-32GF model...")
    model = regnet_y_32gf(weights=None)  # Don't load imagenet weights, we want HAM-trained state
    model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation
    print("🔍 Running evaluation on ISIC test set...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            print(f"   Processing batch {i+1}/{len(test_loader)}...")
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    print("📊 Calculating metrics...")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4)

    # Save model again (checkpoint after ISIC evaluation)
    print("💾 Saving model checkpoint...")
    torch.save(model.state_dict(), model_save_path)

    # Save confusion matrix
    print("🖼️ Saving confusion matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap="Blues")
    plt.title("Confusion Matrix - RegNetY-32GF on ISIC")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # Save results
    print("📄 Writing results to file...")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Final Test Accuracy on ISIC: {acc:.4f}\n\n")
        f.write("Classification Report (precision, recall, f1, support):\n")
        f.write(report + "\n")

        # No training loop here, but logging format placeholder:
        f.write("\nNote: Per-epoch train/test losses are only available from HAM training.\n")

    print("\n✅ Evaluation complete. Results saved at:")
    print(f"   - Model: {model_save_path}")
    print(f"   - Confusion Matrix: {conf_matrix_path}")
    print(f"   - Results: {results_txt_path}")

if __name__ == "__main__":
    main()
