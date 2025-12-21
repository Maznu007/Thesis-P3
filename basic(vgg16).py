"""
BASELINE FL (Standard FedAvg) with VGG16 (weights=None) model structure

Everything remains same as your baseline pipeline:
- Same data loading per client
- Same training loops (local epochs, global epochs)
- Same Standard FedAvg (NO mask awareness)
- Same evaluation + save report + confusion matrix images

Only change:
- Model switched to VGG16 (weights=None) structure similar to your isic_vgg16_final.py
- Shared weights are the "feature_extractor" (vgg features + avgpool + classifier[:6])
- Client head is Linear(4096 -> num_classes) per client

NOTE:
This VGG16 version does NOT include SE-dropout, so it will not demonstrate SE-dropout
zero-dilution by itself. It is purely the VGG16 FL baseline with Standard FedAvg.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
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
OUTPUT_FILE = "baseline_vgg16_standard_fedavg.txt"

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
# VGG16 MODEL (weights=None) WITH FL-SHARED FEATURE EXTRACTOR
# ==============================================================================

class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 feature extractor following your isic_vgg16_final.py spirit:
    - weights=None (random init)
    - Use VGG16 features + avgpool + classifier[:6] to output 4096-dim vector
    """
    def __init__(self):
        super().__init__()
        backbone = models.vgg16(weights=None)  # random initialization

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # classifier[:6] gives: [Linear(25088->4096), ReLU, Dropout, Linear(4096->4096), ReLU, Dropout]
        self.pre_classifier = nn.Sequential(*list(backbone.classifier.children())[:6])

        self.out_features = 4096

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_classifier(x)
        return x


class VGG16ClassificationHead(nn.Module):
    """Client-specific head: Linear(4096 -> num_classes)"""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class VGG16FLClient(nn.Module):
    """Complete client model: shared feature extractor + local head."""
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = VGG16FeatureExtractor()
        self.classifier_head = VGG16ClassificationHead(self.feature_extractor.out_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.feature_extractor(x)
        out = self.classifier_head(feats)
        return out


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
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss


# ==============================================================================
# STANDARD FEDERATED AVERAGING (NO MASK-AWARENESS)
# ==============================================================================

def standard_fed_avg(client_weights, client_sizes):
    """
    Standard FedAvg:
    global_param = sum_i (n_i / sum_j n_j) * client_param_i
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


def get_shared_weights(model):
    """
    Only federate the shared feature extractor:
    - model.feature_extractor.*
    """
    return {k: v.cpu() for k, v in model.state_dict().items()
            if k.startswith('feature_extractor')}


def set_shared_weights(model, shared_weights):
    """
    Load shared weights into client model feature extractor only.
    Keep local head untouched.
    """
    model_state = model.state_dict()
    updated_state = {}
    for k in model_state.keys():
        if k.startswith('feature_extractor') and k in shared_weights:
            updated_state[k] = shared_weights[k].to(model_state[k].device)
        else:
            updated_state[k] = model_state[k]
    model.load_state_dict(updated_state)


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

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
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

    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(all_labels) else 0.0
    avg_loss = total_loss / len(all_labels) if len(all_labels) else 0.0

    return accuracy, f1, avg_loss, all_labels, all_preds


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_baseline_federated_learning():
    print(f"\n{'='*80}")
    print("BASELINE: VGG16 (weights=None) with Standard FedAvg")
    print("Shared: feature_extractor (VGG features + avgpool + classifier[:6])")
    print("Local: Linear(4096 -> num_classes)")
    print(f"{'='*80}\n")

    print(f"Device: {DEVICE}")
    print(f"Aggregation: Standard FedAvg (NO mask-awareness)")
    print(f"{'='*80}\n")

    # Keep transforms same as your FL baseline (augmentation for train)
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

        model = VGG16FLClient(num_classes=num_classes).to(DEVICE)
        client_models[client_id] = model

        print(f"   -> Model: VGG16(weights=None), num_classes={num_classes}")

    if not client_models:
        print("No clients loaded. Exiting.")
        return

    print(f"\nLoaded {len(client_models)} clients\n")

    # Initialize global model weights from first client
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)

    fl_history = defaultdict(lambda: defaultdict(list))

    print(f"{'='*80}")
    print("Training with Standard FedAvg")
    print(f"{'='*80}\n")

    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\nGlobal Epoch {g_epoch}/{GLOBAL_EPOCHS}")
        print("-"*80)

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

        print("  Aggregating: Standard FedAvg ...")
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
            print("  No test loader.")
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" BASELINE: VGG16 (weights=None) with Standard FedAvg\n")
        f.write(" Shared: feature_extractor (VGG features + avgpool + classifier[:6])\n")
        f.write(" Local: Linear(4096 -> num_classes)\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write("  Model: VGG16 (weights=None)\n")
        f.write("  Aggregation: Standard FedAvg\n")
        f.write("-"*80 + "\n\n")

        # Training history table
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

        # Final results + confusion matrices
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
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Reds',
                    xticklabels=metrics['classes'],
                    yticklabels=metrics['classes']
                )
                plt.title(f"BASELINE VGG16 - {client_id.upper()} (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                cm_path = f"baseline_vgg16_confusion_{client_id}.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()

                f.write(f"  -> Saved: {cm_path}\n\n")
            except Exception as e:
                f.write(f"  -> Error saving plot: {e}\n\n")

    print("Baseline Complete!")
    print(f"Results: {OUTPUT_FILE}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_baseline_federated_learning()
