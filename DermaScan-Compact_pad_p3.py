"""
DermaScan-Compact: Efficient CNN with Attention for PAD-UFES-20
Simplified architecture optimized for small datasets

Key Changes:
- Uses pretrained ResNet34 (proven, not too deep)
- SE attention blocks for feature refinement
- Much simpler than multi-scale inception
- Optimized for PAD's 6 classes and ~1600 training samples
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
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


# ============ DermaScan-Compact Model ============
class DermaScanCompact(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DermaScanCompact, self).__init__()
        
        # Load pretrained ResNet34 (balanced depth)
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
            resnet = resnet34(weights=weights)
        else:
            resnet = resnet34(weights=None)
        
        # Extract feature layers (remove FC layer)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Add SE attention after each major layer
        self.se1 = SEBlock(64, reduction=8)
        self.se2 = SEBlock(128, reduction=8)
        self.se3 = SEBlock(256, reduction=16)
        self.se4 = SEBlock(512, reduction=16)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simplified classifier (prevent overfitting)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers with SE attention
        x = self.layer1(x)
        x = self.se1(x)
        
        x = self.layer2(x)
        x = self.se2(x)
        
        x = self.layer3(x)
        x = self.se3(x)
        
        x = self.layer4(x)
        x = self.se4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
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
    print("🚀 Starting DermaScan-Compact training on PAD-UFES-20...")
    
    # Paths
    base_path = r"D:\dataset\pad\organized_pad"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    model_save_path = os.path.join(base_path, "dermascan_compact_model.pth")
    results_txt_path = os.path.join(base_path, "dermascan_results.txt")
    conf_matrix_path = os.path.join(base_path, "dermascan_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Minimal, clinically-realistic augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)
    num_classes = len(train_dataset.classes)
    
    print(f"✅ Classes ({num_classes}): {train_dataset.classes}")
    print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Class weights for imbalance
    class_counts = Counter([label for _, label in train_dataset.samples])
    print(f"   Class distribution: {dict(class_counts)}")
    
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) 
                                  for i in range(num_classes)], dtype=torch.float32).to(device)
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Weighted sampler
    sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    print("🧠 Building DermaScan-Compact model...")
    model = DermaScanCompact(num_classes=num_classes, pretrained=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    
    # Different learning rates for backbone vs new layers
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'layer' in n], 'lr': 1e-5},  # ResNet layers
        {'params': model.se1.parameters(), 'lr': 1e-4},
        {'params': model.se2.parameters(), 'lr': 1e-4},
        {'params': model.se3.parameters(), 'lr': 1e-4},
        {'params': model.se4.parameters(), 'lr': 1e-4},
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
        
        # Training
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
        
        # Validation
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
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title(f"DermaScan-Compact Confusion Matrix\nAcc={final_acc:.2%}, F1={final_f1:.4f}")
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
        f.write("DermaScan-Compact Model Results\n")
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