"""
PURE BASELINE: Standard FedAvg (Full Model Averaging)
======================================================

This is the simplest possible FedAvg baseline:
- NO SE-Attention dropout
- NO mask-aware aggregation  
- NO partial averaging (feature extractor only)
- Averages THE ENTIRE MODEL (backbone + classifier)

This is your TRUE baseline for comparison.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from collections import Counter, defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LOCAL_EPOCHS = 4
GLOBAL_EPOCHS = 5
OUTPUT_FILE = "pure_baseline_full_fedavg_results.txt"

CLIENT_DATASETS = {
    'ham': {
        'train': r"D:\dataset\ham\organized_ham\train",
        'test': r"D:\dataset\ham\organized_ham\test"
    },
    'isic': {
        'train': r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Train",
        'test': r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Test"
    },
    'pad': {
        'train': r"D:\dataset\pad\organized_pad\train",
        'test': r"D:\dataset\pad\organized_pad\test"
    }
}

# ==============================================================================
# MODEL ARCHITECTURE (NO SE DROPOUT, SIMPLE)
# ==============================================================================

class SEBlock(nn.Module):
    """Standard SE-Attention block (NO dropout)."""
    
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


class DermaNetFeatureExtractor(nn.Module):
    """Feature extractor with standard SE-Attention (NO dropout in SE)."""
    
    def __init__(self, pretrained=True):
        super(DermaNetFeatureExtractor, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        self.in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Standard SE block (NO dropout)
        self.se_block = SEBlock(self.in_features, reduction=16)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.unsqueeze(-1).unsqueeze(-1)
        features = self.se_block(features)
        features = features.view(features.size(0), -1)
        return features


class ClassificationHead(nn.Module):
    """Classification head with standard dropout (for regularization)."""
    
    def __init__(self, in_features, num_classes):
        super(ClassificationHead, self).__init__()
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
        return self.classifier(x)


class DermaNetClient(nn.Module):
    """Complete model (will be fully averaged)."""
    
    def __init__(self, num_classes, pretrained=True):
        super(DermaNetClient, self).__init__()
        self.feature_extractor = DermaNetFeatureExtractor(pretrained=pretrained)
        self.classifier_head = ClassificationHead(
            self.feature_extractor.in_features, 
            num_classes
        )
        self.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier_head(features)
        return output


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
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
        else:
            return focal_loss


# ==============================================================================
# PURE FEDERATED AVERAGING (ENTIRE MODEL)
# ==============================================================================

def get_all_weights(model):
    """
    Extract ALL model weights (not just feature extractor).
    This is different from your other codes that only extract feature_extractor.
    """
    return {k: v.cpu() for k, v in model.state_dict().items()}


def set_all_weights(model, weights):
    """
    Set ALL model weights (entire model).
    """
    model.load_state_dict({k: v.to(list(model.parameters())[0].device) 
                           for k, v in weights.items()})


def pure_fed_avg(client_weights, client_sizes):
    """
    Pure FedAvg: Average EVERYTHING (backbone + classifier).
    
    This is the most basic FedAvg implementation:
    - Weighted average by dataset size
    - No partial averaging
    - No mask awareness
    - No special handling
    """
    if not client_weights:
        return None
    
    total_samples = sum(client_sizes.values())
    first_client = list(client_weights.keys())[0]
    global_weights = {}
    
    # Average every single parameter
    for param_name in client_weights[first_client].keys():
        aggregated = torch.zeros_like(
            client_weights[first_client][param_name],
            dtype=torch.float
        )
        
        # Weighted average
        for client_id, weights in client_weights.items():
            client_weight = client_sizes[client_id] / total_samples
            aggregated += weights[param_name].float() * client_weight
        
        global_weights[param_name] = aggregated
    
    return global_weights


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_client_data(data_path, transform, batch_size=32, is_train=True):
    """Load dataset."""
    try:
        dataset = ImageFolder(data_path, transform=transform)
        num_classes = len(dataset.classes)
        
        if is_train:
            class_counts = Counter([label for _, label in dataset.samples])
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for _, label in dataset.samples]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
        
        print(f"   -> Loaded {len(dataset)} samples, {num_classes} classes")
        return loader, num_classes, dataset.classes
    except Exception as e:
        print(f"   !!! Error: {e}")
        return None, 0, []


def train_client(model, train_loader, criterion, local_epochs, device):
    """Standard local training."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epoch_losses = []
    epoch_accs = []
    
    for epoch in range(local_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
    
    return get_all_weights(model), np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate_model(model, test_loader, device):
    """Evaluate model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / len(all_labels)
    
    return accuracy, f1, avg_loss, all_labels, all_preds


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pure_baseline_federated_learning():
    """
    Pure baseline FedAvg implementation.
    Averages ENTIRE model (backbone + classifier) across all clients.
    """
    
    print(f"\n{'='*80}")
    print("📊 PURE BASELINE: Standard Full FedAvg")
    print("   (Simplest possible FedAvg - averages entire model)")
    print(f"{'='*80}\n")
    print(f"Device: {DEVICE}")
    print(f"Global Epochs: {GLOBAL_EPOCHS}")
    print(f"Local Epochs: {LOCAL_EPOCHS}")
    print(f"Aggregation: Pure FedAvg (entire model averaged)")
    print(f"SE Dropout: NO (standard SE-Attention)")
    print(f"Classification Head: Averaged (unlike your other codes)")
    print(f"{'='*80}\n")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize clients
    client_data = {}
    client_models = {}
    client_test_data = {}
    client_train_samples = {}
    
    print("📚 Loading Client Datasets...")
    
    for client_id, paths in CLIENT_DATASETS.items():
        print(f"\n🔹 Client: {client_id.upper()}")
        
        train_loader, num_classes, class_names = load_client_data(
            paths['train'], train_transform, BATCH_SIZE, is_train=True
        )
        
        if train_loader is None:
            print(f"   ⚠️  Skipping {client_id}")
            continue
        
        client_data[client_id] = {
            'train_loader': train_loader,
            'num_classes': num_classes,
            'classes': class_names
        }
        client_train_samples[client_id] = len(train_loader.dataset)
        
        test_loader, _, _ = load_client_data(
            paths['test'], test_transform, BATCH_SIZE, is_train=False
        )
        client_test_data[client_id] = test_loader
        
        model = DermaNetClient(num_classes=num_classes, pretrained=True).to(DEVICE)
        client_models[client_id] = model
    
    if not client_models:
        print("❌ No clients loaded. Aborting.")
        return
    
    print(f"\n✅ Successfully loaded {len(client_models)} clients: {', '.join(client_models.keys()).upper()}\n")
    
    # Initialize global model
    initial_model = list(client_models.values())[0]
    global_weights = get_all_weights(initial_model)
    
    # Training history
    fl_history = defaultdict(lambda: defaultdict(list))
    
    print(f"{'='*80}")
    print("🧠 Starting Pure FedAvg Training")
    print(f"{'='*80}\n")
    
    # Federated training loop
    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\n{'─'*80}")
        print(f"Global Epoch {g_epoch}/{GLOBAL_EPOCHS}")
        print(f"{'─'*80}")
        
        client_updates = {}
        
        # Local training on each client
        for client_id, model in client_models.items():
            print(f"\n  🔹 {client_id.upper()}: Local training...")
            
            # Distribute global model (ENTIRE model)
            set_all_weights(model, global_weights)
            
            # Train locally
            client_weights, loss, acc = train_client(
                model=model,
                train_loader=client_data[client_id]['train_loader'],
                criterion=FocalLoss(alpha=1, gamma=2).to(DEVICE),
                local_epochs=LOCAL_EPOCHS,
                device=DEVICE
            )
            
            client_updates[client_id] = client_weights
            fl_history[client_id]['local_loss'].append(loss)
            fl_history[client_id]['local_acc'].append(acc)
            
            print(f"     Loss: {loss:.4f} | Acc: {acc:.4f}")
        
        # Server aggregation (Pure FedAvg - entire model)
        print(f"\n  🔄 Server: Pure FedAvg aggregation (entire model)...")
        global_weights = pure_fed_avg(client_updates, client_train_samples)
        
        # Record metrics
        avg_loss = np.mean([fl_history[c]['local_loss'][-1] for c in client_data.keys()])
        fl_history['server']['global_loss'].append(avg_loss)
        
        print(f"  ✓ Global epoch complete | Avg loss: {avg_loss:.4f}")
    
    # Final evaluation
    print(f"\n\n{'='*80}")
    print("🔍 FINAL EVALUATION")
    print(f"{'='*80}\n")
    
    final_metrics = {}
    
    for client_id, model in client_models.items():
        print(f"📊 Evaluating {client_id.upper()} Test Set:")
        
        # Apply final global weights (ENTIRE model)
        set_all_weights(model, global_weights)
        
        test_loader = client_test_data[client_id]
        if test_loader is None:
            continue
        
        acc, f1, loss, true_labels, preds = evaluate_model(model, test_loader, DEVICE)
        
        final_metrics[client_id] = {
            'acc': acc,
            'f1': f1,
            'loss': loss,
            'classes': client_data[client_id]['classes'],
            'true_labels': true_labels,
            'predictions': preds
        }
        
        print(f"  Test Loss: {loss:.4f}")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test F1-Score: {f1:.4f}\n")
    
    # Save results
    print(f"💾 Saving results to {OUTPUT_FILE}...\n")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("="*80 + "\n")
        f.write(" PURE BASELINE: Standard Full FedAvg Results\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  Clients: {', '.join(client_data.keys()).upper()}\n")
        f.write(f"  Aggregation: Pure FedAvg (ENTIRE model averaged)\n")
        f.write(f"  SE Dropout: NO (standard SE-Attention)\n")
        f.write(f"  Classifier: AVERAGED (not kept local)\n")
        f.write(f"  Note: This is the simplest baseline for comparison\n")
        f.write("-"*80 + "\n\n")
        
        # Training history
        f.write("Training History:\n")
        f.write("-"*80 + "\n")
        
        header = f"| {'Epoch':<8} | {'Avg Loss':<10} |"
        for cid in client_data.keys():
            header += f" {cid.upper()} Loss | {cid.upper()} Acc |"
        f.write(header + "\n")
        f.write("|" + "-"*78 + "|\n")
        
        for ep in range(GLOBAL_EPOCHS):
            line = f"| {ep+1:<8} | {fl_history['server']['global_loss'][ep]:<10.4f} |"
            for cid in client_data.keys():
                line += f" {fl_history[cid]['local_loss'][ep]:<10.4f} |"
                line += f" {fl_history[cid]['local_acc'][ep]:<9.4f} |"
            f.write(line + "\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Final test results
        f.write("Final Test Results:\n")
        f.write("-"*80 + "\n\n")
        
        for client_id, metrics in final_metrics.items():
            f.write(f"Client: {client_id.upper()}\n")
            f.write(f"{'~'*40}\n")
            f.write(f"  Test Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Test Accuracy: {metrics['acc']:.4f}\n")
            f.write(f"  Test F1-Score: {metrics['f1']:.4f}\n\n")
            
            # Classification report
            report = classification_report(
                metrics['true_labels'],
                metrics['predictions'],
                target_names=metrics['classes'],
                zero_division=0
            )
            f.write("  Classification Report:\n")
            f.write(report + "\n")
            
            # Confusion matrix
            cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
            f.write("\n  Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            
            # Save confusion matrix plot
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                           xticklabels=metrics['classes'],
                           yticklabels=metrics['classes'])
                plt.title(f"Pure Baseline - {client_id.upper()} (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                cm_file = f"pure_baseline_confusion_{client_id}.png"
                plt.savefig(cm_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                f.write(f"  → Confusion matrix saved: {cm_file}\n\n")
            except Exception as e:
                f.write(f"  → Could not save confusion matrix: {e}\n\n")
            
            f.write("-"*80 + "\n\n")
    
    print(f"✅ Pure Baseline Training Complete!")
    print(f"   Results saved: {OUTPUT_FILE}")
    print(f"   Confusion matrices saved as PNG files")
    print(f"\n   Use this as your baseline for comparison!")
    print(f"   Your SE-dropout methods should be compared against this.\n")
    print(f"{'='*80}\n")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*80)
    print(" PURE BASELINE: FULL FEDAVG")
    print(" No SE dropout | No mask-awareness | Full model averaging")
    print("="*80)
    
    run_pure_baseline_federated_learning()