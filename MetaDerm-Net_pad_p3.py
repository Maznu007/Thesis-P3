# metaderm_net_pad.py
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

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class MetaDermNet(nn.Module):
    def __init__(self, num_classes=6, growth_rate=32, block_config=(4, 8, 16, 12)):
        super(MetaDermNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Dense blocks
        num_features = 64
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = nn.Sequential()
            for j in range(num_layers):
                layer = DenseBlock(num_features + j * growth_rate, growth_rate)
                block.add_module(f'dense_{i}_{j}', layer)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            
            # Transition layer (except last)
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.trans_layers.append(trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.bn_final = nn.BatchNorm2d(num_features)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        
        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[i](x)
            if i < len(self.trans_layers):
                x = self.trans_layers[i](x)
        
        x = F.relu(self.bn_final(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def main():
    print("🚀 Starting MetaDerm-Net training on PAD-UFES-20...")
    
    # Paths
    base_path = r"D:\dataset\pad\organized_pad"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    model_save_path = os.path.join(base_path, "metaderm_net_model.pth")
    results_txt_path = os.path.join(base_path, "metaderm_net_results.txt")
    conf_matrix_path = os.path.join(base_path, "metaderm_net_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Enhanced transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
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
    model = MetaDermNet(num_classes=num_classes).to(device)
    
    # Enhanced training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                          patience=3, factor=0.5)
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 10 == 0:
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
        
        # Learning rate scheduling
        scheduler.step(test_acc)
        
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
    plt.title("MetaDerm-Net Confusion Matrix - PAD-UFES-20")
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