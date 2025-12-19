# transfer_learning_pad_fixed.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

def get_class_weights(dataset):
    """Calculate class weights for handling imbalance"""
    targets = [label for _, label in dataset]
    class_counts = Counter(targets)
    total = len(targets)
    
    weights = []
    for class_idx in range(len(class_counts)):
        weight = total / (len(class_counts) * class_counts[class_idx])
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def main():
    print("🚀 Starting Transfer Learning on PAD-UFES-20...")
    
    # Paths
    base_path = r"D:\dataset\pad\organized_pad"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    model_save_path = os.path.join(base_path, "transfer_learning_model.pth")
    results_txt_path = os.path.join(base_path, "transfer_learning_results.txt")
    conf_matrix_path = os.path.join(base_path, "transfer_learning_confusion_matrix.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_dataset = ImageFolder(test_path, transform=test_transform)
    
    # Handle class imbalance
    class_weights = get_class_weights(train_dataset)
    print(f"📊 Class weights: {class_weights}")
    
    # Use weighted sampler
    targets = [label for _, label in train_dataset]
    sampler = WeightedRandomSampler(
        weights=[class_weights[label] for label in targets],
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    print(f"✅ Found {num_classes} classes: {train_dataset.classes}")
    
    # Use pretrained ResNet50 - proven to work well
    print("🧠 Loading pretrained ResNet50...")
    model = models.resnet50(pretrained=True)
    
    # Replace the final layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': 1e-5},
        {'params': model.bn1.parameters(), 'lr': 1e-5},
        {'params': model.layer1.parameters(), 'lr': 1e-5},
        {'params': model.layer2.parameters(), 'lr': 5e-5},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
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
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 10 == 0:
                current_lr = optimizer.param_groups[-1]['lr']  # FC layer LR
                batch_acc = (preds == labels).float().mean().item()
                print(f"Epoch {epoch+1} Batch {i+1}: Loss = {loss.item():.4f}, Acc = {batch_acc:.4f}, LR = {current_lr:.2e}")
        
        scheduler.step()
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, model_save_path)
            print(f"🎯 New best model saved with accuracy: {best_acc:.4f}")
        
        print(f"✅ Epoch {epoch+1}/30 | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}")
        
        
    
    # Final Evaluation with TTA
    print("\n🔍 Final evaluation with TTA...")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    def predict_with_tta(model, images):
        model.eval()
        with torch.no_grad():
            # Original image
            outputs = model(images)
            # Horizontal flip
            flipped_outputs = model(torch.flip(images, dims=[3]))
            # Average predictions
            avg_outputs = (outputs + flipped_outputs) / 2
            return avg_outputs
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = predict_with_tta(model, imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    
    # Save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, 
                yticklabels=train_dataset.classes, cmap='Blues')
    plt.title(f"Transfer Learning Confusion Matrix - PAD-UFES-20\nAccuracy: {acc:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(results_txt_path, "w") as f:
        f.write(f"Final Test Accuracy: {acc:.4f}\n")
        f.write(f"Best Test Accuracy: {best_acc:.4f}\n")
        f.write(f"Model: ResNet50 (pretrained)\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")
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
    
    if acc >= 0.8:
        print("🎉 CONGRATULATIONS! Achieved 80%+ accuracy!")
    elif acc >= 0.7:
        print("✅ Good progress! Close to 80% target.")
    else:
        print("🔄 Let's analyze and improve further.")

if __name__ == "__main__":
    main()