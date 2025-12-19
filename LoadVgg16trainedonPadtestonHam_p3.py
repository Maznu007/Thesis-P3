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
    print("🚀 Starting VGG16 testing (trained on PAD, testing on HAM) with final metrics...")

    # --- Paths and Constants ---
    print("📁 Setting paths and constants...")
    
    # Model Loading Constants
    PAD_NUM_CLASSES = 6  # The VGG16 model was trained and saved with 6 output classes
    
    # Input Paths
    pad_base_dir = r"D:\dataset\pad\organized_pad"
    ham_base_dir = r"D:\dataset\ham\organized_ham"
    
    # Model loading path (using PAD-trained VGG16)
    model_load_path = os.path.join(pad_base_dir, "pad_result_p3", "vgg16onpad_P3", "vgg16_pad_model.pth")
    ham_test_path = os.path.join(ham_base_dir, "test")
    
    # Output Paths
    base_results_dir = os.path.join(ham_base_dir, "ham_results_p3")
    # NEW FOLDER INSIDE ham_results_p3
    results_dir = os.path.join(base_results_dir, "LoadVgg16trainedonPadtestonHam_p3")
    
    os.makedirs(results_dir, exist_ok=True) # Ensure the final results directory exists
    
    results_txt_path = os.path.join(results_dir, "result_vgg16_pad_test_ham.txt")
    conf_matrix_path = os.path.join(results_dir, "confusion_matrix_vgg16_pad_test_ham.png")

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

    # Load HAM Test dataset
    print("📂 Loading HAM Test dataset...")
    test_dataset = ImageFolder(ham_test_path, transform=transform)
    ham_num_classes_actual = len(test_dataset.classes) # Should be 7 classes
    print(f"✅ Loaded {len(test_dataset)} HAM test images with {ham_num_classes_actual} classes.")

    # DataLoaders
    print("📦 Creating DataLoader...")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # HAM Class names (7 classes)
    ham_class_names = test_dataset.classes

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
    print("\n🔍 Starting evaluation on HAM test data...")
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
    # True labels are 0-6 (HAM), but predictions are 0-5 (PAD).
    # We must treat the predictions as categories 0-5 against true classes 0-6.
    
    # The Classification Report and Accuracy will treat any predicted index 6 (which is impossible)
    # as 0 and any true label 6 that is misclassified as an error.
    
    report = classification_report(all_labels, all_preds, target_names=ham_class_names, zero_division=0)
    
    # Accuracy is calculated directly (if pred index == true label index)
    acc = accuracy_score(all_labels, all_preds)

    # CONFUSION MATRIX (Using Original Preds for Visualization):
    print("📊 Calculating confusion matrix...")
    cm_pred_labels = [f"PAD Index {i}" for i in range(PAD_NUM_CLASSES)]
    # CM uses True HAM Classes (rows) vs. Predicted PAD Indices (columns)
    cm_viz = confusion_matrix(all_labels, all_preds, labels=np.arange(ham_num_classes_actual)) 
    
    # Save confusion matrix image
    print("🖼️  Saving confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_viz, annot=True, fmt='d', cmap='Greens',
                xticklabels=cm_pred_labels, yticklabels=ham_class_names)
    plt.title("Confusion Matrix (Trained on PAD, Tested on HAM)")
    plt.xlabel(f"Predicted Class Index (PAD Model Output - {PAD_NUM_CLASSES} indices)")
    plt.ylabel(f"True HAM Class ({ham_num_classes_actual} classes)")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # 4. Save results to result.txt
    print("📄 Writing final results to file...")
    with open(results_txt_path, "w") as f:
        f.write("--- VGG16 (Trained on PAD) Test Results on HAM ---\n\n")
        f.write("NOTE ON CLASS MISMATCH:\n")
        f.write(f"The model was trained on {PAD_NUM_CLASSES} PAD classes, but the test data has {ham_num_classes_actual} HAM classes (0-{ham_num_classes_actual-1}).\n")
        f.write(f"The Classification Report below shows performance against the {ham_num_classes_actual} true HAM classes, where predictions for HAM index 6 are impossible.\n\n")
        
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        
        f.write("Confusion Matrix (True HAM Classes x Predicted PAD Indices):\n")
        f.write(str(cm_viz) + "\n")

    print(f"\n✅ Testing complete. Final results saved in: {results_dir}")

if __name__ == "__main__":
    main()


    #done