# skinnet_a_isic.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SkinNetA(nn.Module):
    def __init__(self, num_classes=9):
        super(SkinNetA, self).__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Residual layers with attention
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.att1 = AttentionGate(64, 64, 32)
        
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x1 = self.layer1(x)
        x1 = self.att1(x1, x1)
        
        x2 = self.layer2(x1)
        x2 = self.att2(x2, x2)
        
        x3 = self.layer3(x2)
        x3 = self.att3(x3, x3)
        
        x4 = self.layer4(x3)
        x4 = self.att4(x4, x4)
        
        x = self.global_pool(x4)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def main():
    print("🚀 Starting SkinNet-A training on ISIC dataset...")
    
    # Paths
    base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    train_path = os.path.join(base_dir, "Train")
    test_path = os.path.join(base_dir, "Test")
    model_save_path = os.path.join(base_dir, "skinnet_a_model.pth")
    results_txt_path = os.path.join(base_dir, "skinnet_a_results.txt")
    conf_matrix_path = os.path.join(base_dir, "skinnet_a_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Transforms with minimal augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    print(f"✅ Found {num_classes} classes: {train_dataset.classes}")
    
    # Model
    model = SkinNetA(num_classes=num_classes).to(device)
    
    # Focal Loss for class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
    
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                                   epochs=20, steps_per_epoch=len(train_loader))
    
    # Training history
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0
    
    print("🏋️ Starting training...")
    
    for epoch in range(20):
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
            scheduler.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 5 == 0:
                print(f"Epoch {epoch+1} Batch {i+1}: Loss = {loss.item():.4f}")
        
        # Epoch metrics
        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        
        # Validation
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)
        
        test_acc = correct_test / total_test
        history["test_acc"].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
        
        print(f"✅ Epoch {epoch+1}/20 | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")
    
    # Final Evaluation
    print("\n🔍 Final evaluation...")
    model.load_state_dict(torch.load(model_save_path))
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
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title("SkinNet-A Confusion Matrix - ISIC")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Save results
    with open(results_txt_path, "w") as f:
        f.write(f"Final Test Accuracy: {acc:.4f}\n")
        f.write(f"Best Test Accuracy: {best_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Training History:\n")
        for i in range(len(history["train_loss"])):
            f.write(f"Epoch {i+1}: Loss={history['train_loss'][i]:.4f}, "
                   f"Train Acc={history['train_acc'][i]:.4f}, "
                   f"Test Acc={history['test_acc'][i]:.4f}\n")
    
    print(f"💾 Model saved to {model_save_path}")
    print(f"📊 Results saved to {results_txt_path}")
    print(f"🖼️ Confusion matrix saved to {conf_matrix_path}")
    print(f"🎯 Final Accuracy: {acc:.4f} | Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()