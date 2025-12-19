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
    print("🚀 Starting VGG16 testing (trained on ISIC, testing on PAD) with final metrics...")

    # --- Paths and Constants ---
    print("📁 Setting paths and constants...")
    
    # Model Loading Constants
    ISIC_NUM_CLASSES = 9  # The VGG16 model was trained and saved with 9 output classes
    
    # Input Paths
    isic_base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    pad_base_dir = r"D:\dataset\pad\organized_pad"
    
    model_load_path = os.path.join(isic_base_dir, "result_vgg16_new", "vgg16_skin_cancer_model.pth")
    pad_test_path = os.path.join(pad_base_dir, "test")
    
    # Output Paths
    base_results_dir = os.path.join(pad_base_dir, "pad_result_p3")
    # NEW FOLDER INSIDE pad_result_p3
    results_dir = os.path.join(base_results_dir, "LoadVgg16trainedonISICtestonPad_p3")
    
    os.makedirs(results_dir, exist_ok=True) # Ensure the final results directory exists
    
    results_txt_path = os.path.join(results_dir, "result_vgg16_isic_test_pad.txt")
    conf_matrix_path = os.path.join(results_dir, "confusion_matrix_vgg16_isic_test_pad.png")

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

    # Load PAD Test dataset
    print("📂 Loading PAD Test dataset...")
    test_dataset = ImageFolder(pad_test_path, transform=transform)
    pad_num_classes_actual = len(test_dataset.classes) # Should be 6 classes
    print(f"✅ Loaded {len(test_dataset)} PAD test images with {pad_num_classes_actual} classes.")

    # DataLoaders
    print("📦 Creating DataLoader...")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # PAD Class names (6 classes)
    pad_class_names = test_dataset.classes

    # --- Model Loading and Class Handling ---
    print("🧠 Loading VGG16 model architecture...")
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Reconfigure the classifier's final layer to match the TRAINED ISIC model's output (9 classes)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, ISIC_NUM_CLASSES) 
    
    print(f"💾 Loading trained weights from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    
    model = model.to(device)
    model.eval()

    # --- Testing and Evaluation ---
    print("\n🔍 Starting evaluation on PAD test data...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            # Outputs a vector of size ISIC_NUM_CLASSES (9)
            outputs = model(imgs) 
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # --- Metrics Calculation and Confusion Matrix ---
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # METRICS CALCULATION (Using Clamped Preds): 
    # Clamp predictions (0-8) to the max target index (0-5) for metric calculation.
    clamped_preds = np.clip(all_preds, 0, pad_num_classes_actual - 1)
    
    report = classification_report(all_labels, clamped_preds, target_names=pad_class_names, zero_division=0)
    acc = accuracy_score(all_labels, clamped_preds)

    # CONFUSION MATRIX (Using Original Preds for Visualization):
    print("📊 Calculating confusion matrix...")
    cm_pred_labels = [f"ISIC Index {i}" for i in range(ISIC_NUM_CLASSES)]
    # CM uses True PAD Classes (rows) vs. Predicted ISIC Indices (columns)
    cm_viz = confusion_matrix(all_labels, all_preds, labels=np.arange(pad_num_classes_actual)) 
    
    # Save confusion matrix image
    print("🖼️  Saving confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_viz, annot=True, fmt='d', cmap='Blues',
                xticklabels=cm_pred_labels, yticklabels=pad_class_names)
    plt.title("Confusion Matrix (Trained on ISIC, Tested on PAD)")
    plt.xlabel(f"Predicted Class Index (ISIC Model Output - {ISIC_NUM_CLASSES} indices)")
    plt.ylabel(f"True PAD Class ({pad_num_classes_actual} classes)")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # 4. Save results to result.txt
    print("📄 Writing final results to file...")
    with open(results_txt_path, "w") as f:
        f.write("--- VGG16 (Trained on ISIC) Test Results on PAD ---\n\n")
        f.write("NOTE ON CLASS MISMATCH:\n")
        f.write(f"The model was trained on {ISIC_NUM_CLASSES} ISIC classes, but the test data has {pad_num_classes_actual} PAD classes (0-{pad_num_classes_actual-1}).\n")
        f.write(f"The Classification Report below is calculated by clamping the predicted indices (0-{ISIC_NUM_CLASSES-1}) to the PAD range (0-{pad_num_classes_actual-1}) for metric consistency.\n\n")
        
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        
        f.write("Confusion Matrix (True PAD Classes x Predicted ISIC Indices):\n")
        f.write(str(cm_viz) + "\n")

    print(f"\n✅ Testing complete. Final results saved in: {results_dir}")

if __name__ == "__main__":
    main()



    #done