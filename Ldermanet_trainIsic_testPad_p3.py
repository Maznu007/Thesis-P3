import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================= SE Block =================
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

# ================= DermaNet-Attention =================
class DermaNetAttention(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DermaNetAttention, self).__init__()
        
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
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
        return self.classifier(features)

# ================= Main =================
if __name__ == "__main__":
    # Paths
    pad_test_path = r"D:\dataset\pad\organized_pad\test"
    pad_result_dir = r"D:\dataset\pad\organized_pad"
    model_path = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\isic_p3_result\isic_dermanet_attention_p3\isic_dermanet_attention_model.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # Dataset & DataLoader
    test_dataset = ImageFolder(pad_test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    num_classes = len(test_dataset.classes)
    print(f"PAD classes ({num_classes}): {test_dataset.classes}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = DermaNetAttention(num_classes=num_classes, pretrained=True).to(device)

    # Load ISIC-trained weights (ignore classifier mismatch)
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Loaded ISIC weights (classifier adapted for PAD).")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Metrics
    final_acc = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

    # Save Confusion Matrix
    conf_matrix_path = os.path.join(pad_result_dir, "pad_isic_dermanet_confusion_matrix.png")
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.title(f"ISIC-DermaNet on PAD Confusion Matrix (F1={final_f1:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=300)
    plt.close()

    # Save Results
    results_txt_path = os.path.join(pad_result_dir, "pad_isic_dermanet_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("ISIC-DermaNet on PAD Model Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {final_acc:.4f}\n")
        f.write(f"Test F1-Score: {final_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("="*60 + "\n")

    print(f"✅ PAD Test (ISIC model) complete!")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1-Score: {final_f1:.4f}")
    print(f"Confusion matrix saved at: {conf_matrix_path}")
    print(f"Results saved at: {results_txt_path}")
