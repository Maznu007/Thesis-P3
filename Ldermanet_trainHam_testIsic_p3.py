"""
Cross-Dataset Evaluation: DermaNet-Attention (HAM10000 → ISIC)
Tests a model trained on HAM10000 on the ISIC dataset with class mapping
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from collections import Counter

# ============ Model Architecture (must match training) ============
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DermaNetAttention(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(DermaNetAttention, self).__init__()
        from torchvision.models import efficientnet_b3
        
        self.backbone = efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.se_block = SEBlock(in_features, reduction=16)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.unsqueeze(-1).unsqueeze(-1)
        features = self.se_block(features)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


# ============ Class Mapping Function ============
def create_class_mapping():
    """
    Maps ISIC classes to HAM10000 classes where possible
    
    HAM10000 classes (7):
    - akiec: Actinic keratoses and intraepithelial carcinoma
    - bcc: Basal cell carcinoma
    - bkl: Benign keratosis-like lesions
    - df: Dermatofibroma
    - mel: Melanoma
    - nv: Melanocytic nevi
    - vasc: Vascular lesions
    
    ISIC classes (9):
    - actinic keratosis → akiec
    - basal cell carcinoma → bcc
    - dermatofibroma → df
    - melanoma → mel
    - nevus → nv
    - pigmented benign keratosis → bkl
    - seborrheic keratosis → bkl
    - squamous cell carcinoma → UNMAPPED (not in HAM10000)
    - vascular lesion → vasc
    """
    
    mapping = {
        'actinic keratosis': 'akiec',
        'basal cell carcinoma': 'bcc',
        'dermatofibroma': 'df',
        'melanoma': 'mel',
        'nevus': 'nv',
        'pigmented benign keratosis': 'bkl',
        'seborrheic keratosis': 'bkl',
        'squamous cell carcinoma': 'UNMAPPED',  # Not in HAM10000
        'vascular lesion': 'vasc'
    }
    
    return mapping


def get_ham_class_order():
    """Return expected HAM10000 class order (alphabetical)"""
    return ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


# ============ Main Evaluation Function ============
def main():
    print("=" * 80)
    print("🔬 Cross-Dataset Evaluation: HAM10000-trained Model → ISIC Test Set")
    print("=" * 80)
    
    # ========== Paths ==========
    model_path = r"D:\dataset\ham\organized_ham\ham_results_p3\demanet_result_p3\dermanet_attention_model.pth"
    isic_test_path = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Test"
    output_base = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    
    # Output files
    results_txt = os.path.join(output_base, "HAM_to_ISIC_evaluation_results.txt")
    conf_matrix_png = os.path.join(output_base, "HAM_to_ISIC_confusion_matrix.png")
    class_mapping_json = os.path.join(output_base, "HAM_to_ISIC_class_mapping.json")
    metrics_json = os.path.join(output_base, "HAM_to_ISIC_detailed_metrics.json")
    
    # ========== Device ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # ========== Load ISIC Test Dataset ==========
    print(f"\n📂 Loading ISIC test dataset from: {isic_test_path}")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    isic_dataset = ImageFolder(isic_test_path, transform=test_transform)
    isic_classes = isic_dataset.classes
    print(f"✅ ISIC classes found ({len(isic_classes)}): {isic_classes}")
    print(f"   Total test samples: {len(isic_dataset)}")
    
    # ========== Class Mapping ==========
    print("\n🔄 Creating class mapping (ISIC → HAM10000)...")
    class_mapping = create_class_mapping()
    ham_classes = get_ham_class_order()
    
    # Create index mapping
    isic_to_ham_idx = {}
    unmapped_classes = []
    
    for isic_idx, isic_class in enumerate(isic_classes):
        ham_class = class_mapping.get(isic_class, 'UNMAPPED')
        if ham_class != 'UNMAPPED':
            ham_idx = ham_classes.index(ham_class)
            isic_to_ham_idx[isic_idx] = ham_idx
        else:
            unmapped_classes.append(isic_class)
            isic_to_ham_idx[isic_idx] = -1  # Mark as unmapped
    
    print(f"✅ Mapping created:")
    for isic_class in isic_classes:
        ham_class = class_mapping.get(isic_class, 'UNMAPPED')
        print(f"   {isic_class:30s} → {ham_class}")
    
    if unmapped_classes:
        print(f"\n⚠️  Warning: {len(unmapped_classes)} unmapped class(es): {unmapped_classes}")
        print(f"   These samples will be excluded from evaluation.")
    
    # Save mapping
    mapping_info = {
        'isic_classes': isic_classes,
        'ham_classes': ham_classes,
        'mapping': class_mapping,
        'unmapped_classes': unmapped_classes,
        'total_isic_samples': len(isic_dataset),
        'mapped_samples': sum(1 for _, label in isic_dataset.samples if isic_to_ham_idx[label] != -1)
    }
    
    with open(class_mapping_json, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    print(f"💾 Class mapping saved: {class_mapping_json}")
    
    # ========== Load Model ==========
    print(f"\n🧠 Loading HAM10000-trained model...")
    print(f"   Model path: {model_path}")
    
    num_ham_classes = len(ham_classes)
    model = DermaNetAttention(num_classes=num_ham_classes).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded successfully (7 HAM10000 classes)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    # ========== DataLoader ==========
    test_loader = DataLoader(isic_dataset, batch_size=32, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # ========== Evaluation ==========
    print("\n🔍 Running evaluation on ISIC test set...")
    
    all_preds = []
    all_labels = []
    all_isic_labels = []  # Original ISIC labels
    skipped_count = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, isic_labels) in enumerate(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Map labels and filter unmapped
            for i, isic_label in enumerate(isic_labels.numpy()):
                ham_label = isic_to_ham_idx[isic_label]
                
                if ham_label != -1:  # Only include mapped samples
                    all_preds.append(preds[i])
                    all_labels.append(ham_label)
                    all_isic_labels.append(isic_label)
                else:
                    skipped_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {(batch_idx + 1) * 32}/{len(isic_dataset)} samples...")
    
    print(f"✅ Evaluation complete!")
    print(f"   Evaluated samples: {len(all_preds)}")
    print(f"   Skipped samples (unmapped): {skipped_count}")
    
    # ========== Calculate Metrics ==========
    print("\n📊 Computing metrics...")
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_ham_classes), zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_ham_classes))
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:        {accuracy:.4f}")
    print(f"   F1 (Weighted):   {f1_weighted:.4f}")
    print(f"   F1 (Macro):      {f1_macro:.4f}")
    
    # ========== Save Confusion Matrix ==========
    print("\n📊 Generating confusion matrix...")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ham_classes, yticklabels=ham_classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'HAM10000-trained Model on ISIC Test Set\nAccuracy: {accuracy:.4f} | F1: {f1_weighted:.4f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class (HAM10000)', fontsize=12)
    plt.ylabel('True Class (HAM10000)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(conf_matrix_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Confusion matrix saved: {conf_matrix_png}")
    
    # ========== Detailed Results ==========
    print("\n💾 Saving detailed results...")
    
    # Text report
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Cross-Dataset Evaluation: HAM10000 -> ISIC\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write(f"  Source model: {model_path}\n")
        f.write(f"  Training dataset: HAM10000 (7 classes)\n")
        f.write(f"  Test dataset: ISIC ({len(isic_classes)} classes)\n\n")
        
        f.write("CLASS MAPPING:\n")
        for isic_class in isic_classes:
            ham_class = class_mapping.get(isic_class, 'UNMAPPED')
            f.write(f"  {isic_class:30s} -> {ham_class}\n")
        f.write(f"\n  Unmapped classes: {unmapped_classes if unmapped_classes else 'None'}\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"  Total ISIC samples: {len(isic_dataset)}\n")
        f.write(f"  Evaluated samples: {len(all_preds)}\n")
        f.write(f"  Skipped samples: {skipped_count}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy:        {accuracy:.4f}\n")
        f.write(f"  F1 (Weighted):   {f1_weighted:.4f}\n")
        f.write(f"  F1 (Macro):      {f1_macro:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS METRICS (HAM10000 Classes):\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, cls in enumerate(ham_classes):
            f.write(f"{cls:<10} {precision[i]:>10.4f}  {recall[i]:>10.4f}  "
                   f"{f1[i]:>10.4f}  {support[i]:>8d}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("=" * 80 + "\n")
        f.write("Rows: True labels | Columns: Predicted labels\n\n")
        
        # Header
        f.write(f"{'':>12}")
        for cls in ham_classes:
            f.write(f"{cls:>8}")
        f.write("\n")
        
        # Matrix
        for i, cls in enumerate(ham_classes):
            f.write(f"{cls:>12}")
            for j in range(len(ham_classes)):
                f.write(f"{cm[i, j]:>8d}")
            f.write(f"  (n={support[i]})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION NOTES:\n")
        f.write("=" * 80 + "\n")
        f.write("1. This evaluation tests domain generalization (HAM10000 -> ISIC)\n")
        f.write("2. Different imaging conditions may affect performance\n")
        f.write("3. Unmapped classes (e.g., squamous cell carcinoma) are excluded\n")
        f.write("4. Class imbalance in test set may impact metrics\n")
    
    print(f"💾 Results saved: {results_txt}")
    
    # JSON metrics
    detailed_metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'evaluated_samples': len(all_preds),
            'skipped_samples': skipped_count
        },
        'per_class': {}
    }
    
    for i, cls in enumerate(ham_classes):
        detailed_metrics['per_class'][cls] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    with open(metrics_json, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"💾 Detailed metrics saved: {metrics_json}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print(f"📊 Overall Performance:")
    print(f"   Accuracy:      {accuracy:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    print(f"   F1 (Macro):    {f1_macro:.4f}")
    print(f"\n📁 Output files saved in:")
    print(f"   {output_base}")
    print("\n🎯 Top performing classes:")
    top_f1_idx = np.argsort(f1)[::-1][:3]
    for idx in top_f1_idx:
        if support[idx] > 0:
            print(f"   {ham_classes[idx]:10s} - F1: {f1[idx]:.4f} (n={support[idx]})")
    print("\n⚠️  Challenging classes:")
    low_f1_idx = np.argsort(f1)[:3]
    for idx in low_f1_idx:
        if support[idx] > 0:
            print(f"   {ham_classes[idx]:10s} - F1: {f1[idx]:.4f} (n={support[idx]})")
    print("=" * 80)


if __name__ == "__main__":
    main()