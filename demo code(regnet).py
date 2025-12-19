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
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ==============================================================================
# 1. DROPOUT MASK MANAGEMENT FOR SE (REGNET-Y HAS SE BLOCKS INSIDE)
# ==============================================================================

class SEDropoutMaskManager:
    """
    Manages fixed dropout masks for SE parameters in federated learning.
    Each client gets a unique, deterministic mask that remains fixed across rounds.

    For RegNetY (torchvision), SE blocks exist inside the backbone. Their parameters
    typically appear in state_dict names containing ".se." (and "weight").
    """

    def __init__(self, dropout_rate=0.3):
        self.dropout_rate = dropout_rate
        self.client_masks = {}
        self.client_seeds = {}

    @staticmethod
    def _is_regnet_se_param(param_name: str) -> bool:
        # RegNetY SE params typically contain ".se." in their name.
        # Mask weights only (conv/linear weights). You can also include biases if desired.
        return (".se." in param_name) and ("weight" in param_name)

    def generate_client_mask(self, client_id, model):
        """
        Generates fixed dropout masks for all SE parameters in a client's model.

        Args:
            client_id: Unique identifier
            model: client model (to read named_parameters)

        Returns:
            Dict[str, torch.Tensor] : masks keyed by parameter name
        """
        seed = abs(hash(client_id)) % (2**32)
        self.client_seeds[client_id] = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        masks = {}

        for name, param in model.named_parameters():
            if self._is_regnet_se_param(name):
                mask = (torch.rand(param.shape) > self.dropout_rate).float()

                # safety: ensure at least one element active
                if mask.sum() == 0:
                    mask.view(-1)[0] = 1.0

                masks[name] = mask.to(param.device)
                print(
                    f"      Generated mask for {name}: shape {tuple(param.shape)}, "
                    f"active: {int(mask.sum().item())}/{mask.numel()} "
                    f"({100.0 * mask.sum().item() / mask.numel():.1f}%)"
                )

        self.client_masks[client_id] = masks
        return masks

    def get_client_mask(self, client_id):
        return self.client_masks.get(client_id, {})

    def apply_masks_to_model(self, model, client_id):
        """
        Applies masks by zeroing out dropped weights after updates or on receipt of global model.
        """
        masks = self.get_client_mask(client_id)
        if not masks:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data *= masks[name].to(param.device)

    def get_coverage_stats(self):
        """
        Coverage stats: how many clients keep each SE parameter element active.
        """
        if not self.client_masks:
            return {}

        all_params = set()
        for masks in self.client_masks.values():
            all_params.update(masks.keys())

        coverage = {}
        for param_name in all_params:
            active_counts = []
            for _, masks in self.client_masks.items():
                if param_name in masks:
                    active_counts.append(masks[param_name].detach().cpu().numpy())

            if active_counts:
                total_coverage = np.sum(active_counts, axis=0)
                coverage[param_name] = {
                    "mean_coverage": float(total_coverage.mean()),
                    "min_coverage": float(total_coverage.min()),
                    "max_coverage": float(total_coverage.max()),
                    "zero_coverage_count": int((total_coverage == 0).sum()),
                    "total_params": int(total_coverage.size),
                }

        return coverage


# ==============================================================================
# 2. MODEL (REGNETY-32GF + CLIENT-SPECIFIC CLASSIFICATION HEAD)
# ==============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
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
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class RegNetFeatureExtractor(nn.Module):
    """
    Shared backbone: RegNetY-32GF.
    We replace the final fc with Identity to output feature vector.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            weights = RegNet_Y_32GF_Weights.DEFAULT
            self.backbone = regnet_y_32gf(weights=weights)
        else:
            self.backbone = regnet_y_32gf(weights=None)

        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class RegNetClient(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.feature_extractor = RegNetFeatureExtractor(pretrained=pretrained)
        self.classifier_head = ClassificationHead(self.feature_extractor.in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.classifier_head(feats)


# ==============================================================================
# 3. FOCAL LOSS
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ==============================================================================
# 4. MASK-AWARE FEDERATED AVERAGING (FOR REGNET SE PARAMS)
# ==============================================================================

def fed_avg_with_masks(client_weights, client_sizes, mask_manager: SEDropoutMaskManager):
    """
    Mask-aware aggregation:
      - For SE params (names contain ".se." and "weight"), average only where clients kept them active.
      - For all other params, standard weighted FedAvg.

    client_weights: {client_id: state_dict}
    client_sizes: {client_id: num_samples}
    """
    if not client_weights:
        return None

    total_samples = sum(client_sizes.values())
    first_client_id = list(client_weights.keys())[0]
    global_weights = {}

    param_names = client_weights[first_client_id].keys()

    for param_name in param_names:
        is_se_param = SEDropoutMaskManager._is_regnet_se_param(param_name)

        if is_se_param:
            param_shape = client_weights[first_client_id][param_name].shape
            aggregated = torch.zeros(param_shape, dtype=torch.float)
            weight_sum = torch.zeros(param_shape, dtype=torch.float)

            for client_id, weights in client_weights.items():
                client_factor = client_sizes[client_id] / total_samples
                p = weights[param_name].float()

                masks = mask_manager.get_client_mask(client_id)
                if param_name in masks:
                    m = masks[param_name].detach().cpu().float()
                    aggregated += p * m * client_factor
                    weight_sum += m * client_factor
                else:
                    aggregated += p * client_factor
                    weight_sum += client_factor

            weight_sum = torch.clamp(weight_sum, min=1e-10)
            global_weights[param_name] = aggregated / weight_sum

        else:
            aggregated = torch.zeros_like(
                client_weights[first_client_id][param_name],
                dtype=torch.float
            )
            for client_id, weights in client_weights.items():
                client_factor = client_sizes[client_id] / total_samples
                aggregated += weights[param_name].float() * client_factor

            global_weights[param_name] = aggregated

    return global_weights


# ==============================================================================
# 5. HELPER FUNCTIONS
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
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=4, pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

        print(f"   -> Loaded {len(dataset)} samples with {num_classes} classes.")
        return loader, num_classes, dataset.classes

    except Exception as e:
        print(f"   !!! Error loading data from {data_path}: {e}")
        return None, 0, []


def get_shared_weights(model):
    # Only average the shared backbone/feature extractor
    return {k: v.cpu() for k, v in model.state_dict().items() if k.startswith("feature_extractor")}


def set_shared_weights(model, shared_weights):
    model_state = model.state_dict()
    updated_state = {
        k: shared_weights[k].to(model_state[k].device) if k.startswith("feature_extractor") else model_state[k]
        for k in model_state
    }
    model.load_state_dict(updated_state)


def train_client(model, train_loader, criterion, local_epochs, device,
                 mask_manager=None, client_id=None):
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

            # Apply fixed dropout masks to SE gradients before optimizer step
            if mask_manager and client_id:
                masks = mask_manager.get_client_mask(client_id)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks and param.grad is not None:
                            param.grad *= masks[name].to(param.device)

            optimizer.step()

            # Apply masks to weights after update (zero out dropped SE weights)
            if mask_manager and client_id:
                mask_manager.apply_masks_to_model(model, client_id)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_losses.append(running_loss / total)
        epoch_accs.append(correct / total)

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

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / len(all_labels)

    return acc, f1, avg_loss, all_labels, all_preds


# ==============================================================================
# 6. MAIN PIPELINE (FEDERATED REGNET + FIXED SE MASKS + MASK-AWARE FEDAVG)
# ==============================================================================

def run_federated_learning_regnet_demo():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {DEVICE}")

    BATCH_SIZE = 32
    LOCAL_EPOCHS = 4
    GLOBAL_EPOCHS = 5

    SE_DROPOUT_RATE = 0.3
    OUTPUT_FILE = "fed_regnet_se_fixed_dropout_results.txt"

    # Dataset paths (keep same pattern)
    CLIENT_DATASETS = {
        "ham": {
            "train": r"D:\dataset\ham\organized_ham\train",
            "test":  r"D:\dataset\ham\organized_ham\test",
        },
        "isic": {
            "train": r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Train",
            "test":  r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Test",
        },
        "pad": {
            "train": r"D:\dataset\pad\organized_pad\train",
            "test":  r"D:\dataset\pad\organized_pad\test",
        },
    }

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n" + "=" * 80)
    print("🚀 FEDERATED LEARNING WITH REGNETY-32GF + SE FIXED DROPOUT (MASK-AWARE FEDAVG)")
    print("=" * 80)
    print("Configuration:")
    print(f"  - Global Epochs: {GLOBAL_EPOCHS}")
    print(f"  - Local Epochs:  {LOCAL_EPOCHS}")
    print(f"  - SE Dropout Rate: {SE_DROPOUT_RATE}")
    print(f"  - Aggregation: Mask-Aware FedAvg (SE params)")
    print("=" * 80 + "\n")

    client_data = {}
    client_models = {}
    client_test_data = {}
    client_train_samples = {}

    print("📚 Loading Client Datasets...")

    for client_id, paths in CLIENT_DATASETS.items():
        print(f"\n🔹 Client: {client_id.upper()}")

        train_loader, num_classes, class_names = load_client_data(
            paths["train"], train_transform, BATCH_SIZE, is_train=True
        )
        if train_loader is None:
            print(f"   ⚠️  Skipping client {client_id} due to data loading error.")
            continue

        client_data[client_id] = {
            "train_loader": train_loader,
            "num_classes": num_classes,
            "classes": class_names,
        }
        client_train_samples[client_id] = len(train_loader.dataset)

        test_loader, _, _ = load_client_data(
            paths["test"], test_transform, BATCH_SIZE, is_train=False
        )
        client_test_data[client_id] = test_loader

        model = RegNetClient(num_classes=num_classes, pretrained=True).to(DEVICE)
        client_models[client_id] = model

    if not client_models:
        print("❌ No client models could be initialized. Aborting.")
        return

    print(f"\n✅ Successfully loaded {len(client_models)} clients: {', '.join(client_models.keys()).upper()}")

    # Mask manager
    print(f"\n🎭 Initializing SE Dropout Mask Manager (RegNet)...")
    mask_manager = SEDropoutMaskManager(dropout_rate=SE_DROPOUT_RATE)

    for client_id, model in client_models.items():
        print(f"\n  🔸 Generating fixed masks for Client {client_id.upper()}:")
        mask_manager.generate_client_mask(client_id, model)
        mask_manager.apply_masks_to_model(model, client_id)

    print(f"\n  📊 SE Parameter Coverage Analysis:")
    coverage_stats = mask_manager.get_coverage_stats()
    if not coverage_stats:
        print("    (No SE params detected for masking — check naming or model variant.)")
    else:
        for param_name, stats in coverage_stats.items():
            print(f"    {param_name}:")
            print(f"      Mean coverage: {stats['mean_coverage']:.2f} clients")
            print(f"      Zero coverage: {stats['zero_coverage_count']}/{stats['total_params']} params")

    # Initialize global weights
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)

    fl_history = defaultdict(lambda: defaultdict(list))

    print(f"\n\n🧠 Starting Federated Training for {GLOBAL_EPOCHS} Global Epochs...")
    print("=" * 80 + "\n")

    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\n{'=' * 20} Global Epoch {g_epoch}/{GLOBAL_EPOCHS} {'=' * 20}")

        client_updates = {}

        for client_id, model in client_models.items():
            print(f"\n  🔹 Client {client_id.upper()}: Local Training...")

            set_shared_weights(model, global_weights)
            mask_manager.apply_masks_to_model(model, client_id)

            client_shared_weights, local_loss, local_acc = train_client(
                model=model,
                train_loader=client_data[client_id]["train_loader"],
                criterion=FocalLoss(alpha=1, gamma=2).to(DEVICE),
                local_epochs=LOCAL_EPOCHS,
                device=DEVICE,
                mask_manager=mask_manager,
                client_id=client_id,
            )

            client_updates[client_id] = client_shared_weights
            fl_history[client_id]["local_loss"].append(local_loss)
            fl_history[client_id]["local_acc"].append(local_acc)

            print(f"     ✓ Loss: {local_loss:.4f} | Acc: {local_acc:.4f}")

        print(f"\n  🔄 Server: Performing Mask-Aware Aggregation...")
        global_weights = fed_avg_with_masks(client_updates, client_train_samples, mask_manager)

        avg_loss = np.mean([fl_history[c]["local_loss"][-1] for c in client_data.keys()])
        avg_acc = np.mean([fl_history[c]["local_acc"][-1] for c in client_data.keys()])
        fl_history["server"]["global_loss"].append(float(avg_loss))
        fl_history["server"]["global_acc"].append(float(avg_acc))

        print(f"  ✓ Global Update Complete | Avg Local Loss: {avg_loss:.4f} | Avg Local Acc: {avg_acc:.4f}")

    # Final evaluation
    print(f"\n\n{'=' * 80}")
    print("🔍 FINAL EVALUATION ON TEST SETS")
    print("=" * 80)

    final_metrics = {}

    for client_id, model in client_models.items():
        print(f"\n📊 Evaluating Client {client_id.upper()} Test Set...")

        set_shared_weights(model, global_weights)
        mask_manager.apply_masks_to_model(model, client_id)

        test_loader = client_test_data[client_id]
        if test_loader is None:
            continue

        acc, f1, loss, true_labels, predictions = evaluate_model(model, test_loader, DEVICE)

        final_metrics[client_id] = {
            "acc": acc,
            "f1": f1,
            "loss": loss,
            "classes": client_data[client_id]["classes"],
            "true_labels": true_labels,
            "predictions": predictions,
        }

        print(f"  ✓ Test Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Save results
    print(f"\n\n💾 Saving Results to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" FEDERATED LEARNING RESULTS: RegNetY-32GF + SE Fixed Dropout (Mask-Aware FedAvg)\n")
        f.write("=" * 80 + "\n")
        f.write("Configuration:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs:  {LOCAL_EPOCHS}\n")
        f.write(f"  SE Dropout Rate: {SE_DROPOUT_RATE}\n")
        f.write(f"  Clients: {', '.join(client_data.keys()).upper()}\n")
        f.write("  Aggregation: Mask-Aware Weighted Averaging\n")
        f.write("  Fixed Masks: Applied to RegNet SE parameters ('.se.' weights)\n")
        f.write("-" * 80 + "\n\n")

        f.write("SE Parameter Coverage:\n")
        f.write("-" * 80 + "\n")
        if not coverage_stats:
            f.write("  No SE params detected for masking.\n")
        else:
            for param_name, stats in coverage_stats.items():
                f.write(f"  {param_name}:\n")
                f.write(f"    Mean coverage: {stats['mean_coverage']:.2f} clients\n")
                f.write(f"    Min coverage: {stats['min_coverage']:.2f}\n")
                f.write(f"    Max coverage: {stats['max_coverage']:.2f}\n")
                f.write(f"    Zero coverage: {stats['zero_coverage_count']}/{stats['total_params']} parameters\n")
        f.write("\n" + "=" * 80 + "\n\n")

        f.write("Training History (Per Global Epoch):\n")
        f.write("-" * 80 + "\n")

        header = f"| {'Epoch':<8} | {'GlobalLoss':<10} | {'GlobalAcc':<10} |"
        for cid in client_data.keys():
            header += f" {cid.upper()} Loss | {cid.upper()} Acc |"
        f.write(header + "\n")
        f.write("|" + "-" * (len(header) - 2) + "|\n")

        for e in range(GLOBAL_EPOCHS):
            line = (
                f"| {e+1:<8} | "
                f"{fl_history['server']['global_loss'][e]:<10.4f} | "
                f"{fl_history['server']['global_acc'][e]:<10.4f} |"
            )
            for cid in client_data.keys():
                line += f" {fl_history[cid]['local_loss'][e]:<10.4f} |"
                line += f" {fl_history[cid]['local_acc'][e]:<9.4f} |"
            f.write(line + "\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write("Final Test Results:\n")
        f.write("-" * 80 + "\n")

        for cid, metrics in final_metrics.items():
            f.write(f"\n{'~'*40}\n")
            f.write(f"Client: {cid.upper()}\n")
            f.write(f"{'~'*40}\n")
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
            f.write(str(cm) + "\n")

            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=metrics["classes"],
                    yticklabels=metrics["classes"]
                )
                plt.title(f"Client {cid.upper()} - Confusion Matrix (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()

                cm_path = f"confusion_matrix_{cid}_regnet_se_dropout.png"
                plt.savefig(cm_path, dpi=300, bbox_inches="tight")
                plt.close()

                f.write(f"  → Confusion matrix saved: {cm_path}\n")
            except Exception as e:
                f.write(f"  → Could not save confusion matrix: {e}\n")

    print("\n✅ Federated Learning Complete!")
    print(f"   Results saved to: {OUTPUT_FILE}")
    print("   Confusion matrices saved as PNG files")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    run_federated_learning_regnet_demo()
