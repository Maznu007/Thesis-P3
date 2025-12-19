# DermaHybrid_SwinNeXt.py

import os
import multiprocessing
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets, models

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# FOCAL LOSS
# -------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        # Use CrossEntropy but reduction none so we can apply focal
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# -------------------------------------------------------
# MODEL: DermaHybrid‑SwinNeXt
# -------------------------------------------------------
class DermaHybridSwinNeXt(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super().__init__()

                # ConvNeXt backbone
        try:
            # Use ConvNeXt Base
            self.cnn = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        except AttributeError:
            # fallback: basic convnext
            self.cnn = models.convnext(weights=models.ConvNeXt_Weights.IMAGENET1K_V1)

        # Remove classifier
        cnn_feat_dim = self.cnn.classifier[2].in_features
        self.cnn.classifier = nn.Identity()


        # Swin Transformer backbone
        from torchvision.models import swin_t, Swin_T_Weights
        self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        swin_feat_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()

        # Fusion head
        fusion_dim = cnn_feat_dim + swin_feat_dim
        # attention gating
        self.fusion_attn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Softmax(dim=-1)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    def forward(self, x):
        feat_cnn = self.cnn(x)                  # 4D: [B, C, H, W]
        feat_cnn = torch.flatten(feat_cnn, 1)   # flatten to [B, C*H*W]

        feat_swin = self.swin(x)                # 2D: [B, F]
        
        fused = torch.cat([feat_cnn, feat_swin], dim=1)  # now both 2D
        attn_weights = self.fusion_attn(fused)
        gated = fused * attn_weights
        out = self.classifier(gated)
        return out



# -------------------------------------------------------
# EVALUATION FUNCTION
# -------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, all_preds, all_labels


# -------------------------------------------------------
# BUILD DATALOADERS
# -------------------------------------------------------
def build_dataloaders(TRAIN_DIR, TEST_DIR, batch_size=16):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    targets = [y for _, y in train_dataset.samples]
    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = 0 if os.name == "nt" else multiprocessing.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset, test_dataset, class_counts


# -------------------------------------------------------
# BUILD MODEL, CRITERION, OPTIMIZER, SCHEDULER
# -------------------------------------------------------
def build_model(num_classes, device, class_counts):
    model = DermaHybridSwinNeXt(num_classes=num_classes).to(device)

    # compute class weights for loss
    weight_list = [1.0 / class_counts[i] for i in range(num_classes)]
    weight_tensor = torch.tensor(weight_list, dtype=torch.float).to(device)
    weight_tensor = weight_tensor / weight_tensor.sum() * num_classes

    criterion = FocalLoss(gamma=2.0, weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    return model, criterion, optimizer, scheduler


# -------------------------------------------------------
# TRAIN LOOP
# -------------------------------------------------------
def train_loop(model, train_loader, test_loader, criterion, optimizer, scheduler, device):
    logs = []
    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total

        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        log = (f"Epoch [{epoch}/20] "
               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
               f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(log)
        logs.append(log)

    return logs


# -------------------------------------------------------
# MAIN ENTRY (for Windows multiprocessing safe)
# -------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Base path: change BASE to your ISIC dataset folder (like your previous scripts)
    BASE = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    TRAIN = os.path.join(BASE, "Train")
    TEST = os.path.join(BASE, "Test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader, train_dataset, test_dataset, class_counts = build_dataloaders(TRAIN, TEST)
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print("Classes:", class_names)
    print("Train counts:", class_counts)

    model, criterion, optimizer, scheduler = build_model(num_classes, device, class_counts)
    logs = train_loop(model, train_loader, test_loader, criterion, optimizer, scheduler, device)

    # Final evaluation
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix – DermaHybrid‑SwinNeXt")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(BASE, "cm_dermahybrid_swinnext.png")
    plt.savefig(cm_path)
    plt.close()

    # Classification report
    report = classification_report(labels, preds, target_names=class_names)
    report_path = os.path.join(BASE, "report_dermahybrid_swinnext.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Epoch Logs:\n")
        for l in logs:
            f.write(l + "\n")

    # Save model
    model_path = os.path.join(BASE, "dermahybrid_swinnext.pth")
    torch.save(model.state_dict(), model_path)

    print("Saved Model:", model_path)
    print("Saved Report:", report_path)
    print("Saved Confusion Matrix:", cm_path)
