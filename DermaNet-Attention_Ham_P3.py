"""
DermaNet-Attention: Hybrid CNN with Squeeze-Excitation Attention
for Skin Lesion Classification on HAM10000 Dataset

Architecture:
- EfficientNetB3 backbone (pretrained)
- SE attention modules
- Custom classifier with dropout
- Focal loss to handle class imbalance
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# ============ Squeeze-Excitation Block ============
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


# ============ DermaNet-Attention Model ============
class DermaNetAttention(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DermaNetAttention, self).__init__()
        
        # Load EfficientNetB3 backbone
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add SE attention
        self.se_block = SEBlock(in_features, reduction=16)
        
        # Custom classifier head
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
        # Extract features
        features = self.backbone(x)
        features = features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        
        # Apply SE attention
        features = self.se_block(features)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output


# ============ Focal Loss (handles class imbalance) ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============ Main Training Function ============
def main():
    print("🚀 Starting DermaNet-Attention training on HAM10000...")
    
    # Paths
    base_dir = r"D:\dataset\ham\organized_ham"
    train_path = os.path.join(base_dir, "train")
    test_path = os.path.join(base_dir, "test")
    model_save_path = os.path.join(base_dir, "dermanet_attention_model.pth")
    results_txt_path = os.path.join(base_dir, "dermanet_results.txt")
    conf_matrix_path = os.path.join(base_dir, "dermanet_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Minimal, clinically-realistic augmentation
    # Simulates natural variations in clinical photography:
    # - Small rotations (phone/camera held at slight angles)
    # - Flips (lesion can be photographed from different sides)
    # - Slight brightness/contrast changes (lighting conditions)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Natural photography angle
        transforms.RandomRotation(10),  # Small camera tilt (±10°)
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Lighting variation
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # HAM10000 normalization
    ])
    
    # Test transform: NO augmentation (real clinical scenario)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)
    num_classes = len(train_dataset.classes)
    print(f"✅ Classes ({num_classes}): {train_dataset.classes}")
    print(f"   Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Handle class imbalance with weighted sampling
    class_counts = Counter([label for _, label in train_dataset.samples])
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    print("🧠 Building DermaNet-Attention model...")
    model = DermaNetAttention(num_classes=num_classes, pretrained=True).to(device)
    
    # Loss and Optimizer
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3 )
    
    # Training history
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_f1": []}
    best_f1 = 0.0
    
    # Training loop
    print("🏋️ Starting training...")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        print(f"\n📍 Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 50 == 0:
                print(f"   Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        
        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)
        
        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"   💾 Best model saved (F1: {best_f1:.4f})")
        
        # Learning rate scheduling
        scheduler.step(test_acc)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Final evaluation
    print("\n🔍 Final evaluation...")
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
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title(f"DermaNet-Attention Confusion Matrix (F1={final_f1:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=300)
    plt.close()
    
    # Save results
    with open(results_txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DermaNet-Attention Model Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Final Test Accuracy: {final_acc:.4f}\n")
        f.write(f"Final Test F1-Score: {final_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("=" * 60 + "\n")
        f.write("Per-Epoch Training History:\n")
        f.write("=" * 60 + "\n")
        for i in range(len(history["train_loss"])):
            f.write(f"Epoch {i+1:2d}: Train Loss={history['train_loss'][i]:.4f}, "
                   f"Train Acc={history['train_acc'][i]:.4f}, "
                   f"Test Acc={history['test_acc'][i]:.4f}, "
                   f"Test F1={history['test_f1'][i]:.4f}\n")
    
    print(f"\n✅ Training complete!")
    print(f"   Final Accuracy: {final_acc:.4f}")
    print(f"   Final F1-Score: {final_f1:.4f}")
    print(f"   Model saved: {model_save_path}")
    print(f"   Results saved: {results_txt_path}")


if __name__ == "__main__":
    main()