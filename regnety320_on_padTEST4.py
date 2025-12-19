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
    print("🚀 Starting PAD (overlap-only) evaluation with HAM-trained RegNetY-32GF...")

    # Paths
    base_dir = r"D:\dataset\pad\organized_pad"
    test_path = os.path.join(base_dir, "test")
    model_ckpt_path = r"D:\dataset\ham\organized_ham\ham_regnetBase_result\regnety320_model.pth"
    save_dir = os.path.join(base_dir, "regnety320_on_PAD_overlap_result")
    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, "regnety320_on_PAD_overlap.pth")
    results_txt_path = os.path.join(save_dir, "results_PAD_overlap.txt")
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix_PAD_overlap.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Transforms (same as HAM training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load PAD test dataset (6 classes)
    print("📂 Loading PAD test dataset...")
    test_dataset = ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"✅ Loaded {len(test_dataset)} test images across {len(test_dataset.classes)} classes.")
    print("   PAD classes:", test_dataset.classes)

    # HAM-trained class names
    HAM_CLASSES = [
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "vascular lesion",
    ]

    # Define overlap classes
    OVERLAP_CLASSES = ["actinic keratosis", "basal cell carcinoma", "melanoma", "nevus"]

    # Load HAM-trained model
    print("🧠 Loading HAM-trained RegNetY-32GF model...")
    model = regnet_y_32gf(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(HAM_CLASSES))
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Inference
    print("🔍 Running inference on PAD overlap classes...")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            print(f"   Processing batch {i+1}/{len(test_loader)}...")
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Map PAD → HAM indices (only overlap)
    pad_to_ham = {test_dataset.classes.index(name): HAM_CLASSES.index(name)
                  for name in OVERLAP_CLASSES if name in test_dataset.classes}

    filtered_labels, filtered_preds = [], []
    for t, p in zip(all_labels, all_preds):
        if t in pad_to_ham:  # only keep overlap classes
            filtered_labels.append(pad_to_ham[t])  # true label remapped
            filtered_preds.append(p)               # HAM prediction idx

    # Metrics
    print("📊 Calculating metrics on overlap classes...")
    acc = accuracy_score(filtered_labels, filtered_preds)
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=range(len(HAM_CLASSES)))
    report = classification_report(filtered_labels, filtered_preds,
                                   labels=range(len(HAM_CLASSES)),
                                   target_names=HAM_CLASSES, digits=4)

    # Save model copy
    torch.save(model.state_dict(), model_save_path)

    # Save confusion matrix
    print("🖼️ Saving confusion matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=HAM_CLASSES, yticklabels=HAM_CLASSES, cmap="Blues")
    plt.title("Confusion Matrix - RegNetY-32GF on PAD (Overlap 4 classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # Save results
    print("📄 Writing results to file...")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation of HAM-trained RegNetY-32GF on PAD Overlap Classes ===\n\n")
        f.write(f"Total images in PAD test set: {len(test_dataset)}\n")
        f.write(f"PAD Classes: {test_dataset.classes}\n\n")
        f.write(f"Overlap Classes Used: {OVERLAP_CLASSES}\n\n")
        f.write(f"Accuracy on overlapping 4 classes: {acc:.4f}\n\n")
        f.write("Classification Report (HAM classes only):\n")
        f.write(report + "\n")

    print("\n✅ Evaluation complete. Results saved at:")
    print(f"   - Model: {model_save_path}")
    print(f"   - Confusion Matrix: {conf_matrix_path}")
    print(f"   - Results: {results_txt_path}")

if __name__ == "__main__":
    main()
