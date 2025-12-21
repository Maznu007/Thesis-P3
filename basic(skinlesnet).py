"""
FULL FL BASELINE: SkinLesNet with Standard FedAvg (copy paste and run)

What this does
- 3 clients: HAM, ISIC, PAD (paths same style as your earlier FL code)
- Model: SkinLesNet architecture exactly like you provided
- Federated part: feature_extractor only (shared across clients)
- Local part: classifier_head (kept local because num_classes can differ per client)
- Aggregation: Standard FedAvg (naive, no mask-awareness)
- Saves:
  1) baseline_skinlesnet_standard_fedavg.txt (full logs, reports, confusion matrices)
  2) baseline_skinlesnet_confusion_<client>.png for each client

Notes
- If your datasets do NOT have the same label set and same class order, this is the safe design.
- If your datasets DO have identical classes and order, you can federate the classifier too, but that is risky.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
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
OUTPUT_FILE = "baseline_skinlesnet_standard_fedavg.txt"

CLIENT_DATASETS = {
    "ham": {
        "train": r"D:\dataset\ham\organized_ham\train",
        "test":  r"D:\dataset\ham\organized_ham\test"
    },
    "isic": {
        "train": r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Train",
        "test":  r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Test"
    },
    "pad": {
        "train": r"D:\dataset\pad\organized_pad\train",
        "test":  r"D:\dataset\pad\organized_pad\test"
    }
}

# ==============================================================================
# MODEL: SkinLesNet (same architecture)
# ==============================================================================

class SkinLesNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# FL safe split (shared features, local classifier)
class SkinLesNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(p=0.5),
        )
        self.out_dim = 128 * 14 * 14

    def forward(self, x):
        return self.block(x)

class SkinLesNetClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class SkinLesNetFLClient(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = SkinLesNetFeatureExtractor()
        self.classifier_head = SkinLesNetClassifierHead(self.feature_extractor.out_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier_head(x)
        return x

# ==============================================================================
# STANDARD FEDAVG (NO MASK-AWARENESS)
# ==============================================================================

def standard_fed_avg(client_weights, client_sizes):
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
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print(f"   -> Loaded {len(dataset)} samples, {num_classes} classes")
        return loader, num_classes, dataset.classes, len(dataset)
    except Exception as e:
        print(f"   !!! Error loading {data_path}: {e}")
        return None, 0, [], 0

def get_shared_weights(model):
    # Only federate feature_extractor
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.startswith("feature_extractor")}

def set_shared_weights(model, shared_weights):
    model_state = model.state_dict()
    updated = {}
    for k in model_state.keys():
        if k.startswith("feature_extractor") and k in shared_weights:
            updated[k] = shared_weights[k].to(model_state[k].device)
        else:
            updated[k] = model_state[k]
    model.load_state_dict(updated)

def train_client(model, train_loader, criterion, local_epochs, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

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
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, [], []

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / len(all_labels)

    return accuracy, f1, avg_loss, all_labels, all_preds

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_baseline_federated_learning():
    print("\n" + "=" * 80)
    print("BASELINE: SkinLesNet with Standard FedAvg")
    print("Shared: feature_extractor only")
    print("Local: classifier head only")
    print("Aggregation: Standard FedAvg (naive)")
    print("=" * 80 + "\n")

    print(f"Device: {DEVICE}")
    print(f"Global Epochs: {GLOBAL_EPOCHS}")
    print(f"Local Epochs: {LOCAL_EPOCHS}")
    print("=" * 80 + "\n")

    # Use your SkinLesNet normalization style (0.5 mean/std)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    client_data = {}
    client_models = {}
    client_test_data = {}
    client_train_samples = {}

    print("Loading Clients...")

    for client_id, paths in CLIENT_DATASETS.items():
        print(f"\nClient: {client_id.upper()}")

        train_loader, num_classes, class_names, train_size = load_client_data(
            paths["train"], train_transform, BATCH_SIZE, is_train=True
        )
        if train_loader is None:
            print("   Skipping client due to load failure.")
            continue

        test_loader, _, _, _ = load_client_data(
            paths["test"], test_transform, BATCH_SIZE, is_train=False
        )

        client_data[client_id] = {
            "train_loader": train_loader,
            "num_classes": num_classes,
            "classes": class_names
        }
        client_test_data[client_id] = test_loader
        client_train_samples[client_id] = train_size

        model = SkinLesNetFLClient(num_classes=num_classes).to(DEVICE)
        client_models[client_id] = model

        print(f"   -> Classes ({num_classes}): {class_names}")
        print(f"   -> Train samples: {train_size}")

    if not client_models:
        print("No clients loaded. Exiting.")
        return

    print(f"\nLoaded {len(client_models)} clients successfully.\n")

    # Initialize global weights from first client
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)

    # History
    fl_history = defaultdict(lambda: defaultdict(list))

    print("=" * 80)
    print("Training with Standard FedAvg")
    print("=" * 80 + "\n")

    criterion = nn.CrossEntropyLoss()

    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\nGlobal Round {g_epoch}/{GLOBAL_EPOCHS}")
        print("-" * 80)

        client_updates = {}

        for client_id, model in client_models.items():
            print(f"  {client_id.upper()}: Training...")

            set_shared_weights(model, global_weights)

            client_weights, loss, acc = train_client(
                model=model,
                train_loader=client_data[client_id]["train_loader"],
                criterion=criterion.to(DEVICE),
                local_epochs=LOCAL_EPOCHS,
                device=DEVICE
            )

            client_updates[client_id] = client_weights
            fl_history[client_id]["local_loss"].append(loss)
            fl_history[client_id]["local_acc"].append(acc)

            print(f"     Loss: {loss:.4f} | Acc: {acc:.4f}")

        print("  Aggregating: Standard FedAvg ...")
        global_weights = standard_fed_avg(client_updates, client_train_samples)

        avg_loss = float(np.mean([fl_history[c]["local_loss"][-1] for c in client_data.keys()]))
        fl_history["server"]["global_loss"].append(avg_loss)
        print(f"  Round complete | Avg loss: {avg_loss:.4f}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80 + "\n")

    final_metrics = {}

    for client_id, model in client_models.items():
        print(f"{client_id.upper()} Test Set:")

        set_shared_weights(model, global_weights)

        test_loader = client_test_data.get(client_id, None)
        if test_loader is None:
            print("  No test loader.")
            continue

        acc, f1, loss, true_labels, preds = evaluate_model(model, test_loader, DEVICE)

        final_metrics[client_id] = {
            "acc": acc,
            "f1": f1,
            "loss": loss,
            "classes": client_data[client_id]["classes"],
            "true_labels": true_labels,
            "predictions": preds
        }

        print(f"  Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n")

    # Save report + confusion matrices
    print(f"Saving results to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" BASELINE: SkinLesNet with Standard FedAvg\n")
        f.write(" Shared: feature_extractor only\n")
        f.write(" Local: classifier head only\n")
        f.write(" Aggregation: Standard FedAvg (naive)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Device: {DEVICE}\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write("-" * 80 + "\n\n")

        f.write("Training History:\n")
        f.write("-" * 80 + "\n")

        header = f"| {'Round':<8} | {'Avg Loss':<10} |"
        for cid in client_data.keys():
            header += f" {cid.upper()} Loss | {cid.upper()} Acc |"
        f.write(header + "\n")
        f.write("|" + "-" * 78 + "|\n")

        for ep in range(GLOBAL_EPOCHS):
            line = f"| {ep+1:<8} | {fl_history['server']['global_loss'][ep]:<10.4f} |"
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
                metrics["true_labels"],
                metrics["predictions"],
                target_names=metrics["classes"],
                zero_division=0
            )
            f.write("  Classification Report:\n")
            f.write(report + "\n")

            cm = confusion_matrix(metrics["true_labels"], metrics["predictions"])
            f.write("\n  Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")

            # Save confusion matrix image
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=metrics["classes"],
                    yticklabels=metrics["classes"]
                )
                plt.title(f"SkinLesNet FL Confusion Matrix: {client_id.upper()} (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                cm_path = f"baseline_skinlesnet_confusion_{client_id}.png"
                plt.savefig(cm_path, dpi=300, bbox_inches="tight")
                plt.close()

                f.write(f"  -> Saved confusion matrix image: {cm_path}\n\n")
            except Exception as e:
                f.write(f"  -> Error saving confusion matrix image: {e}\n\n")

    print("Done.")
    print(f"Text report: {OUTPUT_FILE}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_baseline_federated_learning()
