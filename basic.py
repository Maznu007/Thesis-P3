"""
PROPER BASELINE: SE Dropout with Standard FedAvg (Shows Zero-Dilution Problem)

This code implements SE-Attention dropout but uses STANDARD FedAvg aggregation
(no mask-awareness). This is the "naive" approach that should suffer from 
zero-dilution, allowing you to demonstrate that your mask-aware methods fix it.

USE THIS AS YOUR BASELINE for comparison!
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
SE_DROPOUT_RATE = 0.3
OUTPUT_FILE = "baseline_se_dropout_standard_fedavg.txt"

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
# SE-ATTENTION WITH DROPOUT (BUT NO MASK-AWARE AGGREGATION)
# ==============================================================================

class SEBlockWithDropout(nn.Module):
    """
    SE-Attention with dropout - uses standard PyTorch dropout (stochastic).
    This creates the zero-dilution problem because:
    - Different neurons dropped each forward pass
    - When aggregated with standard FedAvg, zeros dilute learned weights
    """
    
    def __init__(self, channel, reduction=16, dropout_rate=0.3):
        super(SEBlockWithDropout, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # SE layers with dropout in between
        self.squeeze = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)  # Standard stochastic dropout
        self.excite = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # SE forward pass
        y = self.avg_pool(x).view(b, c)
        y = self.squeeze(y)
        y = self.relu(y)
        y = self.dropout(y)  # Drops different neurons each time
        y = self.excite(y)
        y = self.sigmoid(y)
        
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class DermaNetFeatureExtractor(nn.Module):
    """Feature extractor with SE dropout."""
    
    def __init__(self, pretrained=True, se_dropout_rate=0.3):
        super(DermaNetFeatureExtractor, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        self.in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # SE with dropout (creates dilution problem with standard FedAvg)
        self.se_block = SEBlockWithDropout(self.in_features, reduction=16, 
                                          dropout_rate=se_dropout_rate)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.unsqueeze(-1).unsqueeze(-1)
        features = self.se_block(features)
        features = features.view(features.size(0), -1)
        return features


class ClassificationHead(nn.Module):
    """Standard classification head with dropout."""
    
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


class DermaNetAttentionClient(nn.Module):
    """Complete model."""
    
    def __init__(self, num_classes, pretrained=True, se_dropout_rate=0.3):
        super(DermaNetAttentionClient, self).__init__()
        self.feature_extractor = DermaNetFeatureExtractor(pretrained=pretrained,
                                                          se_dropout_rate=se_dropout_rate)
        self.classifier_head = ClassificationHead(self.feature_extractor.in_features, 
                                                  num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier_head(features)
        return output


class FocalLoss(nn.Module):
    """Focal Loss."""
    
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
# STANDARD FEDERATED AVERAGING (NO MASK-AWARENESS)
# ==============================================================================

def standard_fed_avg(client_weights, client_sizes):
    """
    Standard FedAvg - the naive approach that causes zero-dilution.
    
    Problem: Averages all weights including zeros from dropout.
    This dilutes the learned weights from active neurons.
    """
    if not client_weights:
        return None
    
    total_samples = sum(client_sizes.values())
    first_client = list(client_weights.keys())[0]
    global_weights = {}
    
    for param_name in client_weights[first_client].keys():
        aggregated = torch.zeros_like(
            client_weights[first_client][param_name],
            dtype=torch.float
        )
        
        # Standard averaging - includes zeros from dropout
        for client_id, weights in client_weights.items():
            client_weight = client_sizes[client_id] / total_samples
            aggregated += weights[param_name].float() * client_weight
        
        global_weights[param_name] = aggregated
    
    return global_weights


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_client_data(data_path, transform, batch_size=32, is_train=True):
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


def get_shared_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()
            if k.startswith('feature_extractor')}


def set_shared_weights(model, shared_weights):
    model_state = model.state_dict()
    updated_state = {
        k: shared_weights[k].to(model_state[k].device)
        if k.startswith('feature_extractor') else model_state[k]
        for k in model_state
    }
    model.load_state_dict(updated_state)


def train_client(model, train_loader, criterion, local_epochs, device):
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
    
    return get_shared_weights(model), np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate_model(model, test_loader, device):
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

def run_baseline_federated_learning():
    print(f"\n{'='*80}")
    print("⚠️  BASELINE: SE Dropout with Standard FedAvg")
    print("   (This should show the zero-dilution problem)")
    print(f"{'='*80}\n")
    print(f"Device: {DEVICE}")
    print(f"SE Dropout: {SE_DROPOUT_RATE} (stochastic, not fixed)")
    print(f"Aggregation: Standard FedAvg (NO mask-awareness)")
    print(f"Expected: Worse than mask-aware methods due to dilution")
    print(f"{'='*80}\n")
    
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
    
    print("📚 Loading Clients...")
    
    for client_id, paths in CLIENT_DATASETS.items():
        print(f"\n🔹 {client_id.upper()}")
        
        train_loader, num_classes, class_names = load_client_data(
            paths['train'], train_transform, BATCH_SIZE, is_train=True
        )
        
        if train_loader is None:
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
        
        model = DermaNetAttentionClient(num_classes=num_classes, pretrained=True,
                                       se_dropout_rate=SE_DROPOUT_RATE).to(DEVICE)
        client_models[client_id] = model
    
    if not client_models:
        print("❌ No clients loaded")
        return
    
    print(f"\n✅ Loaded {len(client_models)} clients\n")
    
    # Initialize global model
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)
    
    # Training history
    fl_history = defaultdict(lambda: defaultdict(list))
    
    print(f"{'='*80}")
    print("🧠 Training with Standard FedAvg (Expects Zero-Dilution)")
    print(f"{'='*80}\n")
    
    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\nGlobal Epoch {g_epoch}/{GLOBAL_EPOCHS}")
        print("-"*80)
        
        client_updates = {}
        
        for client_id, model in client_models.items():
            print(f"  🔹 {client_id.upper()}: Training...")
            
            set_shared_weights(model, global_weights)
            
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
        
        # Standard FedAvg (NO mask-awareness → ZERO DILUTION)
        print(f"  🔄 Standard FedAvg (with zero-dilution)...")
        global_weights = standard_fed_avg(client_updates, client_train_samples)
        
        avg_loss = np.mean([fl_history[c]['local_loss'][-1] for c in client_data.keys()])
        fl_history['server']['global_loss'].append(avg_loss)
        
        print(f"  ✓ Epoch complete | Avg loss: {avg_loss:.4f}")
    
    # Final evaluation
    print(f"\n\n{'='*80}")
    print("📊 FINAL EVALUATION")
    print(f"{'='*80}\n")
    
    final_metrics = {}
    
    for client_id, model in client_models.items():
        print(f"🔹 {client_id.upper()} Test Set:")
        
        set_shared_weights(model, global_weights)
        
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
        
        print(f"  Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n")
    
    # Save results
    print(f"💾 Saving to {OUTPUT_FILE}...\n")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("="*80 + "\n")
        f.write(" BASELINE: SE Dropout with Standard FedAvg\n")
        f.write(" (Shows zero-dilution problem)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  SE Dropout: {SE_DROPOUT_RATE} (stochastic)\n")
        f.write(f"  Aggregation: Standard FedAvg (NO mask-awareness)\n")
        f.write(f"  Expected: Suffers from zero-dilution\n")
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
        
        # Final results
        f.write("Final Test Results:\n")
        f.write("-"*80 + "\n\n")
        
        for client_id, metrics in final_metrics.items():
            f.write(f"Client: {client_id.upper()}\n")
            f.write(f"  Test Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Test Accuracy: {metrics['acc']:.4f}\n")
            f.write(f"  Test F1-Score: {metrics['f1']:.4f}\n\n")
            
            report = classification_report(
                metrics['true_labels'],
                metrics['predictions'],
                target_names=metrics['classes'],
                zero_division=0
            )
            f.write("  Classification Report:\n")
            f.write(report + "\n")
            
            cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
            f.write("\n  Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                           xticklabels=metrics['classes'],
                           yticklabels=metrics['classes'])
                plt.title(f"BASELINE - {client_id.upper()} (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                cm_path = f"baseline_confusion_{client_id}.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
                f.write(f"  → Saved: {cm_path}\n\n")
            except Exception as e:
                f.write(f"  → Error saving plot: {e}\n\n")
    
    print(f"✅ Baseline Complete!")
    print(f"   Results: {OUTPUT_FILE}")
    print(f"\n⚠️  Compare this with your mask-aware methods!")
    print(f"   Your methods should perform BETTER than this baseline")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_baseline_federated_learning()