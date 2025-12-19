import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import regnet_y_32gf
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("🚀 Starting ISIC (9-class) evaluation with HAM-trained RegNetY-32GF...")

    # Paths
    print("📁 Setting paths...")
    base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    test_path = os.path.join(base_dir, "Test")  # Your ISIC test folder (9 classes)
    model_ckpt_path = r"D:\dataset\ham\organized_ham\ham_regnetBase_result\regnety320_model.pth"
    save_dir = os.path.join(base_dir, "regnety320_on_ISIC_9class_result")
    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, "regnety320_on_ISIC_9class.pth")
    results_txt_path = os.path.join(save_dir, "results_ISIC_9class.txt")
    pred_csv_path = os.path.join(save_dir, "predictions_ISIC_9class.csv")
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix_ISIC_9class.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Transforms (same as HAM training)
    print("🌀 Preparing image transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load ISIC test dataset (9 classes)
    print("📂 Loading ISIC test dataset...")
    test_dataset = ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"✅ Loaded {len(test_dataset)} test images across {len(test_dataset.classes)} classes.")
    print("   Classes found:", test_dataset.classes)

    # HAM-trained class names (7 only)
    HAM_CLASSES = [
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "vascular lesion",
    ]

    # Load HAM-trained model
    print("🧠 Loading HAM-trained RegNetY-32GF model...")
    model = regnet_y_32gf(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(HAM_CLASSES))
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Inference
    print("🔍 Running inference on ISIC test set (9-class)...")
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            print(f"   Processing batch {i+1}/{len(test_loader)}...")
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Save predictions CSV
    print("💾 Saving raw predictions to CSV...")
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label_idx", "true_label_name",
                         "pred_label_idx", "pred_label_name"] + HAM_CLASSES)
        for i, (true_idx, pred_idx, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
            true_name = test_dataset.classes[true_idx]
            pred_name = HAM_CLASSES[pred_idx]
            writer.writerow([i, true_idx, true_name, pred_idx, pred_name] +
                            [f"{p:.6f}" for p in prob])

    # Metrics (only for overlapping HAM classes)
    print("📊 Calculating metrics for overlapping classes (⚠️ only HAM 7)...")

    # Map ISIC class indices -> HAM class indices
    isic_to_ham = {test_dataset.classes.index(name): HAM_CLASSES.index(name)
                   for name in HAM_CLASSES if name in test_dataset.classes}

    filtered_labels, filtered_preds = [], []
    for t, p in zip(all_labels, all_preds):
        if t in isic_to_ham:  # keep only HAM-known classes
            filtered_labels.append(isic_to_ham[t])  # map ISIC idx -> HAM idx
            filtered_preds.append(p)                # HAM prediction idx already correct

    if filtered_labels:
        acc = accuracy_score(filtered_labels, filtered_preds)
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=range(len(HAM_CLASSES)))
        report = classification_report(filtered_labels, filtered_preds,
                                       labels=range(len(HAM_CLASSES)),
                                       target_names=HAM_CLASSES, digits=4)
    else:
        acc, cm, report = None, None, None

    # Save model copy
    print("💾 Saving checkpoint copy...")
    torch.save(model.state_dict(), model_save_path)

    # Save confusion matrix
    if cm is not None:
        print("🖼️ Saving confusion matrix...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=HAM_CLASSES, yticklabels=HAM_CLASSES, cmap="Blues")
        plt.title("Confusion Matrix - RegNetY-32GF on ISIC (HAM 7 classes)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(conf_matrix_path)
        plt.close()

    # Save results
    print("📄 Writing results to file...")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Evaluation of HAM-trained RegNetY-32GF on ISIC 9-class dataset ===\n\n")
        f.write(f"Total images: {len(test_dataset)}\n")
        f.write(f"ISIC Classes: {test_dataset.classes}\n\n")
        if acc is not None:
            f.write(f"Accuracy on overlapping HAM 7 classes: {acc:.4f}\n\n")
            f.write("Classification Report (HAM 7 classes only):\n")
            f.write(report + "\n")
        else:
            f.write("⚠️ Metrics skipped: no overlap with HAM classes found.\n")

        f.write("\nPredictions saved to CSV with probabilities for analysis.\n")

    print("\n✅ Evaluation complete. Results saved at:")
    print(f"   - Model: {model_save_path}")
    print(f"   - Predictions CSV: {pred_csv_path}")
    if cm is not None:
        print(f"   - Confusion Matrix: {conf_matrix_path}")
    print(f"   - Results: {results_txt_path}")

if __name__ == "__main__":
    main()
