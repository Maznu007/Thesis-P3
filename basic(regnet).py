"""
PROPER BASELINE (RegNet Version): SE Dropout with Standard FedAvg (Shows Zero-Dilution Problem)

You requested:
- Replace the DermaNet/EfficientNet model structure with RegNetY-32GF structure
- Keep everything else the same:
  - SE attention with stochastic dropout
  - Standard FedAvg (no mask-awareness)
  - Same FL pipeline, loaders, loss, evaluation, saving
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from collections import Counter, defaultdict
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
OUTPUT_FILE = "baseline_se_dropout_standard_fedavg_regnet.txt"

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
# SE-ATTENTION WITH STOCHASTIC DROPOUT (NAIVE BASELINE)
# ==============================================================================

class SEBlockWithDropout(nn.Module):
    """
    SE-Attention with standard (stochastic) dropout.
    This is intentionally naive for the baseline.
    """
    def __init__(self, channel, reduction=16, dropout_rate=0.3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.squeeze = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.excite = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.squeeze(y)
        y = self.relu(y)
        y = self.dropout(y)   # stochastic
        y = self.excite(y)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

# ==============================================================================
# REGNET CLIENT MODEL (REPLACES DERMANET/EFFICIENTNET)
# ==============================================================================

class RegNetSEClient(nn.Module):
    """
    RegNetY-32GF backbone + SE dropout + simple fc classifier (like your RegNet code).
    """
    def __init__(self, num_classes, pretrained=True, se_dropout_rate=0.3):
        super().__init__()

        if pretrained:
            weights = RegNet_Y_32GF_Weights.DEFAULT
            self.backbone = regnet_y_32gf(weights=weights)
        else:
            self.backbone = regnet_y_32gf(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.se = SEBlockWithDropout(
            channel=in_features,
            reduction=16,
            dropout_rate=se_dropout_rate
        )

        # RegNet-style classifier head
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)                 # [B, C]
        x = x.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        x = self.se(x)
        x = x.view(x.size(0), -1)            # [B, C]
        x = self.fc(x)                       # [B, num_classes]
        return x

# ==============================================================================
# LOSS
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

# ==============================================================================
# STANDARD FEDAVG (NO MASK AWARENESS)
# ==============================================================================

def standard_fed_avg(client_weights, client_sizes):
    """
    Standard FedAvg (naive baseline).
    """
    if not client_weights:
        return None

    total_samples = sum(client_sizes.values())
    first_client = list(client_weights.keys())[0]
    global_weights = {}

    for param_name in client_weights[first_client].keys():
        aggregated = torch.zeros_like(client_weights[first_client][param_name], dtype=torch.float)

        for client_id, weights in client_weights.items():
            w = client_sizes[client_id] / total_samples
            aggregated += weights[param_name].float() * w

        global_weights[param_name] = aggregated

    return global_weights

# ==============================================================================
# DATA LOADING
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

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        print(f"   -> Loaded {len(dataset)} samples, {num_classes} classes")
        return loader, num_classes, dataset.classes
    except Exception as e:
        print(f"   !!! Error: {e}")
        return None, 0, []

# ==============================================================================
# FEDERATED SHARE/SET WEIGHTS
# ==============================================================================

def get_shared_weights(model):
    """
    Federate: backbone + SE only.
    Keep classifier fc local.
    """
    sd = model.state_dict()
    return {k: v.cpu() for k, v in sd.items() if k.startswith("backbone") or k.startswith("se")}

def set_shared_weights(model, shared_weights):
    sd = model.state_dict()
    for k, v in shared_weights.items():
        if k in sd:
            sd[k] = v.to(sd[k].device)
    model.load_state_dict(sd)

# ==============================================================================
# TRAIN/EVAL
# ==============================================================================

def train_client(model, train_loader, criterion, local_epochs, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epoch_losses = []
    epoch_accs = []

    for _ in range(local_epochs):
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

    return get_shared_weights(model), float(np.mean(epoch_losses)), float(np.mean(epoch_accs))

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
# MAIN FL PIPELINE
# ==============================================================================

def run_baseline_federated_learning():
    print(f"\n{'='*80}")
    print("BASELINE: RegNetY-32GF + SE Dropout with Standard FedAvg")
    print("(Naive baseline expected to suffer under stochastic dropout)")
    print(f"{'='*80}\n")

    print(f"Device: {DEVICE}")
    print(f"Backbone: RegNetY-32GF")
    print(f"SE Dropout: {SE_DROPOUT_RATE} (stochastic)")
    print("Aggregation: Standard FedAvg (NO mask-awareness)")
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

    client_data = {}
    client_models = {}
    client_test_data = {}
    client_train_samples = {}

    print("Loading Clients...")

    for client_id, paths in CLIENT_DATASETS.items():
        print(f"\nClient: {client_id.upper()}")

        train_loader, num_classes, class_names = load_client_data(
            paths['train'], train_transform, BATCH_SIZE, is_train=True
        )
        if train_loader is None:
            continue

        test_loader, _, _ = load_client_data(
            paths['test'], test_transform, BATCH_SIZE, is_train=False
        )

        client_data[client_id] = {
            'train_loader': train_loader,
            'num_classes': num_classes,
            'classes': class_names
        }
        client_test_data[client_id] = test_loader
        client_train_samples[client_id] = len(train_loader.dataset)

        model = RegNetSEClient(
            num_classes=num_classes,
            pretrained=True,
            se_dropout_rate=SE_DROPOUT_RATE
        ).to(DEVICE)

        client_models[client_id] = model

    if not client_models:
        print("No clients loaded. Check dataset paths.")
        return

    print(f"\nLoaded {len(client_models)} clients.\n")

    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)

    fl_history = defaultdict(lambda: defaultdict(list))

    print(f"{'='*80}")
    print("Training with Standard FedAvg")
    print(f"{'='*80}\n")

    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\nGlobal Epoch {g_epoch}/{GLOBAL_EPOCHS}")
        print("-" * 80)

        client_updates = {}

        for client_id, model in client_models.items():
            print(f"  {client_id.upper()}: Training...")

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

        print("  Aggregating with Standard FedAvg...")
        global_weights = standard_fed_avg(client_updates, client_train_samples)

        avg_loss = float(np.mean([fl_history[c]['local_loss'][-1] for c in client_data.keys()]))
        fl_history['server']['global_loss'].append(avg_loss)
        print(f"  Epoch complete | Avg loss: {avg_loss:.4f}")

    # Final evaluation
    print(f"\n\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}\n")

    final_metrics = {}

    for client_id, model in client_models.items():
        print(f"{client_id.upper()} Test Set:")

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
    print(f"Saving to {OUTPUT_FILE}...\n")

    with open(OUTPUT_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" BASELINE: RegNetY-32GF + SE Dropout with Standard FedAvg\n")
        f.write(" (Naive baseline expected to suffer without mask-awareness)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write("  Backbone: RegNetY-32GF\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  SE Dropout: {SE_DROPOUT_RATE} (stochastic)\n")
        f.write("  Aggregation: Standard FedAvg (NO mask-awareness)\n")
        f.write("-" * 80 + "\n\n")

        f.write("Training History:\n")
        f.write("-" * 80 + "\n")
        header = f"| {'Epoch':<8} | {'Avg Loss':<10} |"
        for cid in client_data.keys():
            header += f" {cid.upper()} Loss | {cid.upper()} Acc |"
        f.write(header + "\n")
        f.write("|" + "-" * 78 + "|\n")

        for ep in range(GLOBAL_EPOCHS):
            line = f"| {ep + 1:<8} | {fl_history['server']['global_loss'][ep]:<10.4f} |"
            for cid in client_data.keys():
                line += f" {fl_history[cid]['local_loss'][ep]:<10.4f} |"
                line += f" {fl_history[cid]['local_acc'][ep]:<9.4f} |"
            f.write(line + "\n")

        f.write("\n" + "=" * 80 + "\n\n")

        f.write("Final Test Results:\n")
        f.write("-" * 80 + "\n\n")

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
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=metrics['classes'],
                    yticklabels=metrics['classes']
                )
                plt.title(f"BASELINE - {client_id.upper()} (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                cm_path = f"baseline_confusion_regnet_{client_id}.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
                f.write(f"  -> Saved: {cm_path}\n\n")
            except Exception as e:
                f.write(f"  -> Error saving plot: {e}\n\n")

    print("Baseline Complete.")
    print(f"Results: {OUTPUT_FILE}")
    print("Compare with your mask-aware methods. They should perform better.")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_baseline_federated_learning()
