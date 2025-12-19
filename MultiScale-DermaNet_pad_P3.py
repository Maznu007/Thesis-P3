"""
MultiScale-DermaNet: Multi-Scale Feature Extraction Network
for Skin Lesion Classification on PAD-UFES-20 Dataset

Architecture:
- Parallel multi-scale convolutional streams
- Inception-style modules with attention
- Feature fusion with learned weights
- Dense connections for gradient flow
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# ============ Channel Attention (Simplified SE) ============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max_val = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg)
        max_out = self.fc(max_val)
        attention = (avg_out + max_out).view(b, c, 1, 1)
        return x * attention


# ============ Multi-Scale Inception Block ============
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 -> 5x5 conv (as two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: MaxPool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention on fused features
        self.attention = ChannelAttention(out_channels)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate all branches
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Apply channel attention
        out = self.attention(out)
        return out


# ============ Dense Block for Feature Reuse ============
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1),
            nn.Dropout2d(0.2)
        )
    
    def forward(self, x):
        new_features = self.conv(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ============ MultiScale-DermaNet Model ============
class MultiScaleDermaNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiScaleDermaNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Multi-scale blocks
        self.ms_block1 = MultiScaleBlock(64, 128)
        self.pool1 = nn.MaxPool2d(2)
        
        # Dense block for feature reuse
        self.dense1 = DenseBlock(128, growth_rate=32, num_layers=4)
        dense1_out = 128 + 32 * 4  # 256
        
        # Transition layer
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(dense1_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dense1_out, 256, kernel_size=1),
            nn.AvgPool2d(2)
        )
        
        self.ms_block2 = MultiScaleBlock(256, 256)
        
        # Dense block 2
        self.dense2 = DenseBlock(256, growth_rate=32, num_layers=4)
        dense2_out = 256 + 32 * 4  # 384
        
        # Transition layer 2
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(dense2_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dense2_out, 384, kernel_size=1),
            nn.AvgPool2d(2)
        )
        
        self.ms_block3 = MultiScaleBlock(384, 512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
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
        
        # Multi-scale + Dense blocks
        x = self.ms_block1(x)
        x = self.pool1(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        
        x = self.ms_block2(x)
        x = self.dense2(x)
        x = self.trans2(x)
        
        x = self.ms_block3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        return x


# ============ Label Smoothing Cross Entropy ============
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        if self.weight is not None:
            true_dist = true_dist * self.weight[target].unsqueeze(1)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ============ Main Training Function ============
def main():
    print("🚀 Starting MultiScale-DermaNet training on PAD-UFES-20...")
    
    # Paths
    base_path = r"D:\dataset\pad\organized_pad"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    model_save_path = os.path.join(base_path, "multiscale_dermanet_model.pth")
    results_txt_path = os.path.join(base_path, "multiscale_results.txt")
    conf_matrix_path = os.path.join(base_path, "multiscale_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Minimal, clinically-realistic augmentation for PAD-UFES-20
    # Reflects natural variations in smartphone clinical photography
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Small camera angle variations
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Natural lighting differences
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # PAD normalization (same as HAM)
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
    print(f"   Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Class weights
    class_counts = Counter([label for _, label in train_dataset.samples])
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) 
                                  for i in range(num_classes)], dtype=torch.float32).to(device)
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Weighted sampler
    sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=24, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    print("🧠 Building MultiScale-DermaNet...")
    model = MultiScaleDermaNet(num_classes=num_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Training history
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_f1": []}
    best_f1 = 0.0
    
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
        
        scheduler.step()
        
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
            torch.save(model.state_dict(), model_save_path)
            print(f"   💾 Best model saved (F1: {best_f1:.4f})")
    
    # Load best model
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
    
    final_acc = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title(f"MultiScale-DermaNet Confusion Matrix (F1={final_f1:.4f})")
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
        f.write("MultiScale-DermaNet Model Results\n")
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
    print(f"   Final Accuracy: {final_acc:.4f}")
    print(f"   Final F1-Score: {final_f1:.4f}")
    print(f"   Results saved to: {results_txt_path}")


if __name__ == "__main__":
    main()