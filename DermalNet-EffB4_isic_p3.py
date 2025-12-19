# DermalNet-EffB4_isic_p3.py
"""
DermalNet-EffB4
Improved ISIC skin cancer classifier, Windows-safe multiprocessing,
EfficientNet-B4 + minimal realistic augmentations + weighted sampling + focal loss.
20 epochs.
"""

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
# FOCAL LOSS IMPLEMENTATION
# -------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


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
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc = 100 * correct / total

    return loss, acc, all_preds, all_labels


# -------------------------------------------------------
# BUILD DATALOADERS
# -------------------------------------------------------
def build_dataloaders(TRAIN_DIR, TEST_DIR, batch_size=16):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
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

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # WINDOWS SAFE
    num_workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset, test_dataset, class_counts


# -------------------------------------------------------
# BUILD MODEL
# -------------------------------------------------------
def build_model(num_classes, device, class_counts):

    try:
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
    except:
        print("Warning: Pretrained weights not found. Using random init.")
        model = models.efficientnet_b4(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    # class weights for loss
    weight_tensor = torch.tensor([1/class_counts[i] for i in range(num_classes)],
                                 dtype=torch.float).to(device)

    weight_tensor = weight_tensor / weight_tensor.sum() * num_classes

    criterion = FocalLoss(gamma=2.0, weight=weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    model = model.to(device)

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
            images, labels = images.to(device), labels.to(device)

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
        train_acc = 100 * correct / total

        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        log = (f"Epoch [{epoch}/20] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
               f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        print(log)
        logs.append(log)

    return logs


# -------------------------------------------------------
# MAIN ENTRY (MANDATORY FOR WINDOWS)
# -------------------------------------------------------
if __name__ == "__main__":

    multiprocessing.freeze_support()

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

    logs = train_loop(model, train_loader, test_loader, criterion,
                      optimizer, scheduler, device)

    # ----- Final Evaluation -----
    _, _, preds, labels = evaluate(model, test_loader, criterion, device)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(BASE, "cm_dermalnet_effb4.png")
    plt.savefig(cm_path)
    plt.close()

    report = classification_report(labels, preds, target_names=class_names)
    report_path = os.path.join(BASE, "result_dermalnet_effb4.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("\nEpoch Logs:\n")
        for l in logs:
            f.write(l + "\n")

    model_path = os.path.join(BASE, "dermalnet_effb4.pth")
    torch.save(model.state_dict(), model_path)

    print("Saved Model:", model_path)
    print("Saved Report:", report_path)
    print("Saved CM:", cm_path)

