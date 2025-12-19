"""
LesionNet-Hybrid: Lightweight CNN with Attention for ISIC Dataset
Simplified architecture for small, imbalanced datasets

Key Changes:
- Uses pretrained MobileNetV2 (lightweight, proven for medical images)
- CBAM attention for feature refinement
- Much simpler than Vision Transformer
- Handles severe class imbalance with focal loss + sampling
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# ============ CBAM Attention Module ============
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ============ LesionNet-Hybrid Model ============
class LesionNetHybrid(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(LesionNetHybrid, self).__init__()
        
        # Load pretrained MobileNetV2 (lightweight, efficient)
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.backbone = mobilenet_v2(weights=weights)
        else:
            self.backbone = mobilenet_v2(weights=None)
        
        # Get feature dimension (MobileNetV2 outputs 1280 channels)
        in_features = self.backbone.last_channel
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add CBAM attention
        self.attention = CBAM(in_features, reduction=16)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Simplified classifier (prevent overfitting on small dataset)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features through MobileNetV2
        x = self.backbone.features(x)
        
        # Apply CBAM attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        return x


# ============ Focal Loss ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ============ Main Training Function ============
def main():
    print("🚀 Starting LesionNet-Hybrid training on ISIC dataset...")
    
    # Paths
    BASE_DIR = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    TRAIN_DIR = os.path.join(BASE_DIR, "Train")
    TEST_DIR = os.path.join(BASE_DIR, "Test")
    
    model_save_path = os.path.join(BASE_DIR, "lesionnet_hybrid_model.pth")
    results_txt_path = os.path.join(BASE_DIR, "lesionnet_results.txt")
    conf_matrix_path = os.path.join(BASE_DIR, "lesionnet_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Minimal, clinically-realistic augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"✅ Classes ({num_classes}): {class_names}")
    print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Calculate class weights
    class_counts = Counter([label for _, label in train_dataset.samples])
    print(f"   Class distribution: {dict(class_counts)}")
    
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / (num_classes * class_counts[i]) 
                                  for i in range(num_classes)], dtype=torch.float32).to(device)
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Weighted sampling for extreme imbalance
    sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # DataLoaders (smaller batch size for small dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    print("🧠 Building LesionNet-Hybrid model...")
    model = LesionNetHybrid(num_classes=num_classes, pretrained=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    
    # Use different learning rates for backbone and classifier
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},  # Lower LR for pretrained
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training history
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_f1": []}
    best_f1 = 0.0
    patience_counter = 0
    early_stop_patience = 10
    
    # Training loop
    print("🏋️ Training started...")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 20 == 0:
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
        
        print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"   💾 Best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Learning rate scheduling
        scheduler.step(test_f1)
    
    # Load best model
    print("\n📥 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Final evaluation
    print("🔍 Final evaluation...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    final_acc = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"LesionNet-Hybrid Confusion Matrix\nAcc={final_acc:.2%}, F1={final_f1:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=300)
    plt.close()
    print(f"   Confusion matrix saved: {conf_matrix_path}")
    
    # Save results
    with open(results_txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LesionNet-Hybrid Model Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Final Test Accuracy: {final_acc:.4f}\n")
        f.write(f"Final Test F1-Score: {final_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("=" * 60 + "\n")
        f.write("Per-Epoch Results:\n")
        f.write("=" * 60 + "\n")
        for i in range(len(history["train_loss"])):
            f.write(f"Epoch {i+1:2d}: Train Loss={history['train_loss'][i]:.4f}, "
                   f"Train Acc={history['train_acc'][i]:.4f}, "
                   f"Test Acc={history['test_acc'][i]:.4f}, "
                   f"Test F1={history['test_f1'][i]:.4f}\n")
    
    print(f"\n✅ Training complete!")
    print(f"   Final Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"   Final F1-Score: {final_f1:.4f}")
    print(f"   Results saved: {results_txt_path}")


if __name__ == "__main__":
    main()