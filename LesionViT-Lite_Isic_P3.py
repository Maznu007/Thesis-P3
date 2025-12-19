"""
LesionViT-Lite: Lightweight Vision Transformer with CBAM Attention
for Skin Lesion Classification on ISIC Dataset

Architecture:
- Custom lightweight ViT with reduced parameters
- CBAM (Convolutional Block Attention Module)
- Hybrid CNN-Transformer approach
- Class-balanced focal loss
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
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


# ============ Patch Embedding ============
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use Conv2d for patch extraction
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


# ============ Transformer Encoder Block ============
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ============ LesionViT-Lite Model ============
class LesionViTLite(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=9, 
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super(LesionViTLite, self).__init__()
        
        # CNN stem with CBAM for low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128),
        )
        
        # Patch embedding (on 56x56 feature map)
        self.patch_embed = PatchEmbedding(img_size=56, patch_size=patch_size, 
                                          in_channels=128, embed_dim=embed_dim)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # CNN stem
        x = self.stem(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use class token for classification
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


# ============ Focal Loss with Class Weights ============
class FocalLossWithWeights(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLossWithWeights, self).__init__()
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
    print("🚀 Starting LesionViT-Lite training on ISIC dataset...")
    
    # Paths
    BASE_DIR = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    TRAIN_DIR = os.path.join(BASE_DIR, "Train")
    TEST_DIR = os.path.join(BASE_DIR, "Test")
    
    model_save_path = os.path.join(BASE_DIR, "lesionvit_lite_model.pth")
    results_txt_path = os.path.join(BASE_DIR, "lesionvit_results.txt")
    conf_matrix_path = os.path.join(BASE_DIR, "lesionvit_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Minimal, clinically-realistic augmentation for ISIC
    # Reflects natural variations in dermatoscopic imaging
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Dermatoscope can be oriented differently
        transforms.RandomRotation(15),  # Small rotations (±15°)
        transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Lighting/skin tone variation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Test transform: NO augmentation (real clinical scenario)
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
    
    # Calculate class weights for imbalance
    class_counts = Counter([label for _, label in train_dataset.samples])
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / (num_classes * class_counts[i]) 
                                  for i in range(num_classes)], dtype=torch.float32).to(device)
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # Weighted sampling
    sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    print("🧠 Building LesionViT-Lite model...")
    model = LesionViTLite(
        img_size=224,
        patch_size=14,  # Smaller patches for better detail
        num_classes=num_classes,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = FocalLossWithWeights(alpha=class_weights, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Training history
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_f1": []}
    best_f1 = 0.0
    
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
        
        scheduler.step()
        
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
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"LesionViT-Lite Confusion Matrix (F1={final_f1:.4f})")
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
        f.write("LesionViT-Lite Model Results\n")
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