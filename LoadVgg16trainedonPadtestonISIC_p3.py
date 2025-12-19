import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("🚀 Starting VGG16 testing (trained on PAD, testing on ISIC) with final metrics...")

    # --- Paths and Constants ---
    print("📁 Setting paths and constants...")
    
    # Model Loading Constants
    PAD_NUM_CLASSES = 6  # The VGG16 model was trained and saved with 6 output classes
    
    # Input Paths
    pad_base_dir = r"D:\dataset\pad\organized_pad"
    isic_base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    
    # Model loading path (using PAD-trained VGG16)
    model_load_path = os.path.join(pad_base_dir, "pad_result_p3", "vgg16onpad_P3", "vgg16_pad_model.pth")
    isic_test_path = os.path.join(isic_base_dir, "Test")
    
    # Output Paths
    base_results_dir = os.path.join(isic_base_dir, "isic_p3_result")
    # NEW FOLDER INSIDE isic_p3_result
    results_dir = os.path.join(base_results_dir, "LoadVgg16trainedonPadtestonISIC_p3")
    
    os.makedirs(results_dir, exist_ok=True) # Ensure the final results directory exists
    
    results_txt_path = os.path.join(results_dir, "result_vgg16_pad_test_isic.txt")
    conf_matrix_path = os.path.join(results_dir, "confusion_matrix_vgg16_pad_test_isic.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # --- Data Loading ---
    print("🌀 Preparing image transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # VGG standard normalization
    ])

    # Load ISIC Test dataset
    print("📂 Loading ISIC Test dataset...")
    test_dataset = ImageFolder(isic_test_path, transform=transform)
    isic_num_classes_actual = len(test_dataset.classes) # Should be 9 classes
    print(f"✅ Loaded {len(test_dataset)} ISIC test images with {isic_num_classes_actual} classes.")

    # DataLoaders
    print("📦 Creating DataLoader...")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # ISIC Class names (9 classes)
    isic_class_names = test_dataset.classes

    # --- Model Loading and Class Handling ---
    print("🧠 Loading VGG16 model architecture...")
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Reconfigure the classifier's final layer to match the TRAINED PAD model's output (6 classes)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, PAD_NUM_CLASSES) 
    
    print(f"💾 Loading trained weights from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    
    model = model.to(device)
    model.eval()

    # --- Testing and Evaluation ---
    print("\n🔍 Starting evaluation on ISIC test data...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            # Outputs a vector of size PAD_NUM_CLASSES (6)
            outputs = model(imgs) 
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- Metrics Calculation and Confusion Matrix ---
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # METRICS CALCULATION: 
    # True labels are 0-8 (ISIC), but predictions are 0-5 (PAD).
    
    # The Classification Report and Accuracy will treat any predicted index > 5 as a misclassification 
    # for the true ISIC classes (indices 6, 7, 8).
    
    report = classification_report(all_labels, all_preds, target_names=isic_class_names, zero_division=0)
    
    # Accuracy is calculated directly (if pred index == true label index)
    acc = accuracy_score(all_labels, all_preds)

    # CONFUSION MATRIX (Using Original Preds for Visualization):
    print("📊 Calculating confusion matrix...")
    cm_pred_labels = [f"PAD Index {i}" for i in range(PAD_NUM_CLASSES)]
    # CM uses True ISIC Classes (rows) vs. Predicted PAD Indices (columns)
    cm_viz = confusion_matrix(all_labels, all_preds, labels=np.arange(isic_num_classes_actual)) 
    
    # Save confusion matrix image
    print("🖼️  Saving confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_viz, annot=True, fmt='d', cmap='Purples',
                xticklabels=cm_pred_labels, yticklabels=isic_class_names)
    plt.title("Confusion Matrix (Trained on PAD, Tested on ISIC)")
    plt.xlabel(f"Predicted Class Index (PAD Model Output - {PAD_NUM_CLASSES} indices)")
    plt.ylabel(f"True ISIC Class ({isic_num_classes_actual} classes)")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # 4. Save results to result.txt
    print("📄 Writing final results to file...")
    with open(results_txt_path, "w") as f:
        f.write("--- VGG16 (Trained on PAD) Test Results on ISIC ---\n\n")
        f.write("NOTE ON CLASS MISMATCH:\n")
        f.write(f"The model was trained on {PAD_NUM_CLASSES} PAD classes, but the test data has {isic_num_classes_actual} ISIC classes (0-{isic_num_classes_actual-1}).\n")
        f.write(f"The Classification Report below shows performance against the {isic_num_classes_actual} true ISIC classes, where predictions for ISIC indices 6, 7, and 8 are impossible.\n\n")
        
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        
        f.write("Confusion Matrix (True ISIC Classes x Predicted PAD Indices):\n")
        f.write(str(cm_viz) + "\n")

    print(f"\n✅ Testing complete. Final results saved in: {results_dir}")

if __name__ == "__main__":
    main()

#done