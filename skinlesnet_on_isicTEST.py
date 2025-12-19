# skinlesnet_on_isicTEST.py
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
# Define SkinLesNet (same as training)
# ===============================
class SkinLesNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesNet, self).__init__()
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
    print("🚀 Evaluating PAD-trained SkinLesNet on ISIC dataset...")

    # ===============================
    # Paths
    # ===============================
    pad_model_path = r"D:\dataset\pad\organized_pad\result_skinlesnet\skinlesnet_model.pth"
    isic_test_path = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Test"
    save_dir       = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\skinlesnet_from_PAD_result"
    os.makedirs(save_dir, exist_ok=True)

    results_txt_path  = os.path.join(save_dir, "results.txt")
    conf_matrix_path  = os.path.join(save_dir, "confusion_matrix.png")

    # ===============================
    # Device
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # ===============================
    # Transforms
    # ===============================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ===============================
    # Load ISIC test dataset
    # ===============================
    isic_dataset = ImageFolder(isic_test_path, transform=transform)
    isic_loader  = DataLoader(isic_dataset, batch_size=32, shuffle=False)
    isic_classes = isic_dataset.classes
    print(f"📂 ISIC classes ({len(isic_classes)}): {isic_classes}")

    # ===============================
    # Define overlap classes
    # ===============================
    overlap_classes = [
        "actinic keratosis",
        "basal cell carcinoma",
        "melanoma",
        "nevus",
        "seborrheic keratosis",
        "squamous cell carcinoma"
    ]

    # PAD model had 6 classes
    pad_classes = overlap_classes

    # Build ISIC → PAD index mapping (only overlap classes)
    isic_to_pad = {isic_classes.index(c): pad_classes.index(c) for c in overlap_classes}
    print("🔄 ISIC → PAD index mapping:", isic_to_pad)

    # ===============================
    # Load PAD-trained model
    # ===============================
    print("🧠 Loading PAD-trained SkinLesNet...")
    model = SkinLesNet(num_classes=len(pad_classes)).to(device)  # 6 outputs
    model.load_state_dict(torch.load(pad_model_path, map_location=device))
    model.eval()
    print("✅ PAD-trained SkinLesNet loaded successfully.")

    # ===============================
    # Inference
    # ===============================
    print("🔍 Running inference on ISIC test set...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in isic_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for t, p in zip(labels.numpy(), preds):
                if t in isic_to_pad:  # only keep overlap classes
                    all_labels.append(isic_to_pad[t])  # true mapped to PAD idx
                    all_preds.append(p)

    # ===============================
    # Metrics
    # ===============================
    print("📊 Calculating metrics on overlap classes...")
    valid_pad_indices = [pad_classes.index(c) for c in overlap_classes]

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=valid_pad_indices)
    report = classification_report(all_labels, all_preds,
                                   labels=valid_pad_indices,
                                   target_names=overlap_classes, digits=4)

    print(f"✅ Accuracy on overlap classes: {acc:.4f}")
    print(report)

    # ===============================
    # Save confusion matrix
    # ===============================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=overlap_classes, yticklabels=overlap_classes,
                cmap="Blues")
    plt.title("Confusion Matrix (PAD-trained SkinLesNet on ISIC Overlap)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # ===============================
    # Save results
    # ===============================
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation of PAD-trained SkinLesNet on ISIC (Overlap Classes) ===\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    print("\n✅ Evaluation complete. Results saved at:")
    print(f"   - Confusion Matrix: {conf_matrix_path}")
    print(f"   - Results: {results_txt_path}")


if __name__ == "__main__":
    main()
