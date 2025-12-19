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
import random

# ==============================================================================
# 1. DROPOUT MASK MANAGEMENT FOR SE-ATTENTION LAYERS
# ==============================================================================

class SEDropoutMaskManager:
    """
    Manages fixed dropout masks for SE-Attention layers in federated learning.
    Each client gets a unique, deterministic mask that remains fixed across rounds.
    """
    
    def __init__(self, dropout_rate=0.3):
        """
        Args:
            dropout_rate: Probability of dropping neurons (0.0-1.0)
        """
        self.dropout_rate = dropout_rate
        self.client_masks = {}  # Stores masks for each client
        self.client_seeds = {}  # Stores seeds for reproducibility
        
    def generate_client_mask(self, client_id, model):
        """
        Generates fixed dropout masks for all SE layers in a client's model.
        
        Args:
            client_id: Unique identifier for the client
            model: The client's model (to extract SE layer dimensions)
            
        Returns:
            Dictionary of masks for each SE layer parameter
        """
        # Create deterministic seed from client_id
        seed = abs(hash(client_id)) % (2**32)
        self.client_seeds[client_id] = seed
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        masks = {}
        
        # Get all SE layer parameters
        for name, param in model.named_parameters():
            if 'se_block.fc' in name and 'weight' in name:
                # Generate binary mask (1 = keep, 0 = drop)
                mask = (torch.rand(param.shape) > self.dropout_rate).float()
                
                # Ensure at least some neurons are active (safety check)
                if mask.sum() == 0:
                    # If all dropped, randomly activate at least one
                    mask.view(-1)[0] = 1.0
                    
                masks[name] = mask.to(param.device)
                print(f"      Generated mask for {name}: shape {param.shape}, "
                      f"active neurons: {mask.sum().item()}/{mask.numel()} "
                      f"({100 * mask.sum().item() / mask.numel():.1f}%)")
        
        self.client_masks[client_id] = masks
        return masks
    
    def get_client_mask(self, client_id):
        """Returns the stored mask for a client."""
        return self.client_masks.get(client_id, {})
    
    def apply_masks_to_model(self, model, client_id):
        """
        Applies the fixed dropout masks to a model's SE layer parameters.
        
        Args:
            model: The model to apply masks to
            client_id: The client whose masks to use
        """
        masks = self.get_client_mask(client_id)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data *= masks[name].to(param.device)
    
    def get_coverage_stats(self):
        """
        Analyzes parameter coverage across all clients.
        Returns statistics about which parameters are trained by how many clients.
        """
        if not self.client_masks:
            return {}
        
        # Get all parameter names
        all_params = set()
        for masks in self.client_masks.values():
            all_params.update(masks.keys())
        
        coverage = {}
        for param_name in all_params:
            # Count how many clients have this parameter active
            active_counts = []
            for client_id, masks in self.client_masks.items():
                if param_name in masks:
                    mask = masks[param_name]
                    active_counts.append(mask.cpu().numpy())
            
            if active_counts:
                # Sum across clients to get coverage per parameter
                total_coverage = np.sum(active_counts, axis=0)
                coverage[param_name] = {
                    'mean_coverage': total_coverage.mean(),
                    'min_coverage': total_coverage.min(),
                    'max_coverage': total_coverage.max(),
                    'zero_coverage_count': (total_coverage == 0).sum(),
                    'total_params': total_coverage.size
                }
        
        return coverage


# ==============================================================================
# 2. MODIFIED SE-BLOCK WITH DROPOUT SUPPORT
# ==============================================================================

class SEBlockWithDropout(nn.Module):
    """
    Squeeze-Excitation Block with support for fixed dropout masks.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlockWithDropout, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.dropout_mask = None  # Will be set externally

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    def set_dropout_mask(self, masks):
        """Sets the dropout masks for this SE block's FC layers."""
        self.dropout_mask = masks


# ==============================================================================
# 3. MODIFIED MODEL ARCHITECTURE
# ==============================================================================

class DermaNetFeatureExtractor(nn.Module):
    """Feature Extractor with SE-Attention and dropout support."""
    
    def __init__(self, pretrained=True):
        super(DermaNetFeatureExtractor, self).__init__()
        
        # Load EfficientNetB3 backbone
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Get feature dimension
        self.in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add SE attention with dropout support
        self.se_block = SEBlockWithDropout(self.in_features, reduction=16)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        
        # Apply SE attention
        features = self.se_block(features)
        features = features.view(features.size(0), -1)
        return features


class ClassificationHead(nn.Module):
    """Classification Head (Client-specific layers)."""
    
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
    """Full Model combining shared and client-specific parts."""
    
    def __init__(self, num_classes, pretrained=True):
        super(DermaNetAttentionClient, self).__init__()
        self.feature_extractor = DermaNetFeatureExtractor(pretrained=pretrained)
        self.classifier_head = ClassificationHead(self.feature_extractor.in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier_head(features)
        return output


# ==============================================================================
# 4. FOCAL LOSS
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
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
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==============================================================================
# 5. MASK-AWARE FEDERATED AVERAGING
# ==============================================================================

def fed_avg_with_masks(client_weights, client_sizes, mask_manager):
    """
    Performs Mask-Aware Federated Averaging.
    
    For SE-Attention parameters:
        - Only average weights from clients where the parameter was active (not dropped)
    For other parameters:
        - Standard weighted averaging
    
    Args:
        client_weights: Dict of {client_id: state_dict}
        client_sizes: Dict of {client_id: num_samples}
        mask_manager: SEDropoutMaskManager instance
        
    Returns:
        Aggregated global weights
    """
    if not client_weights:
        return None

    total_samples = sum(client_sizes.values())
    first_client_id = list(client_weights.keys())[0]
    global_weights = {}
    
    # Get all parameter names
    param_names = client_weights[first_client_id].keys()
    
    for param_name in param_names:
        # Check if this is an SE layer parameter that has masks
        is_se_param = 'se_block.fc' in param_name and 'weight' in param_name
        
        if is_se_param:
            # MASK-AWARE AGGREGATION for SE parameters
            # Initialize accumulator
            param_shape = client_weights[first_client_id][param_name].shape
            aggregated_param = torch.zeros(param_shape, dtype=torch.float)
            weight_sum = torch.zeros(param_shape, dtype=torch.float)
            
            for client_id, weights in client_weights.items():
                param = weights[param_name].float()
                client_weight = client_sizes[client_id] / total_samples
                
                # Get client's mask for this parameter
                client_masks = mask_manager.get_client_mask(client_id)
                if param_name in client_masks:
                    mask = client_masks[param_name].cpu().float()
                    
                    # Only accumulate where mask is active
                    aggregated_param += param * mask * client_weight
                    weight_sum += mask * client_weight
                else:
                    # Fallback to standard averaging if mask not found
                    aggregated_param += param * client_weight
                    weight_sum += client_weight
            
            # Normalize by the sum of weights for each parameter
            # Avoid division by zero
            weight_sum = torch.clamp(weight_sum, min=1e-10)
            global_weights[param_name] = aggregated_param / weight_sum
            
        else:
            # STANDARD AVERAGING for non-SE parameters
            aggregated_param = torch.zeros_like(
                client_weights[first_client_id][param_name], 
                dtype=torch.float
            )
            
            for client_id, weights in client_weights.items():
                client_weight = client_sizes[client_id] / total_samples
                aggregated_param += weights[param_name].float() * client_weight
            
            global_weights[param_name] = aggregated_param
    
    return global_weights


# ==============================================================================
# 6. HELPER FUNCTIONS
# ==============================================================================

def load_client_data(data_path, transform, batch_size=32, is_train=True):
    """Loads dataset and returns DataLoader, number of classes, and class names."""
    try:
        dataset = ImageFolder(data_path, transform=transform)
        num_classes = len(dataset.classes)
        
        if is_train:
            # Handle class imbalance with weighted sampling
            class_counts = Counter([label for _, label in dataset.samples])
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for _, label in dataset.samples]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                              num_workers=4, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)

        print(f"   -> Loaded {len(dataset)} samples with {num_classes} classes.")
        return loader, num_classes, dataset.classes
    except Exception as e:
        print(f"   !!! Error loading data from {data_path}: {e}")
        return None, 0, []


def get_shared_weights(model):
    """Extracts weights for the shared feature extractor layers."""
    return {k: v.cpu() for k, v in model.state_dict().items() 
            if k.startswith('feature_extractor')}


def set_shared_weights(model, shared_weights):
    """Updates the shared feature extractor layers with global weights."""
    model_state = model.state_dict()
    updated_state = {
        k: shared_weights[k].to(model_state[k].device) 
        if k.startswith('feature_extractor') else model_state[k]
        for k in model_state
    }
    model.load_state_dict(updated_state)


def train_client(model, train_loader, criterion, local_epochs, device, 
                mask_manager=None, client_id=None):
    """
    Trains a client model locally with fixed SE dropout masks.
    
    Args:
        model: Client model
        train_loader: Training data loader
        criterion: Loss function
        local_epochs: Number of local training epochs
        device: Device to train on
        mask_manager: SEDropoutMaskManager instance
        client_id: Client identifier
        
    Returns:
        Tuple of (shared_weights, avg_loss, avg_acc)
    """
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
            
            # Apply fixed dropout masks to SE gradients before optimizer step
            if mask_manager and client_id:
                masks = mask_manager.get_client_mask(client_id)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks and param.grad is not None:
                            param.grad *= masks[name].to(param.device)
            
            optimizer.step()
            
            # Apply masks to weights after update (zero out dropped weights)
            if mask_manager and client_id:
                mask_manager.apply_masks_to_model(model, client_id)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
    
    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    
    return get_shared_weights(model), avg_loss, avg_acc


def evaluate_model(model, test_loader, device):
    """
    Evaluates a model on test data.
    
    Returns:
        Tuple of (accuracy, f1_score, loss, true_labels, predictions)
    """
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
# 7. MAIN FEDERATED LEARNING PIPELINE
# ==============================================================================

def run_federated_learning():
    """Main federated learning pipeline with SE-Attention fixed dropout."""
    
    # ========== CONFIGURATION ==========
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {DEVICE}")
    
    BATCH_SIZE = 32
    LOCAL_EPOCHS = 4
    GLOBAL_EPOCHS = 5
    
    SE_DROPOUT_RATE = 0.3  # Dropout rate for SE layers
    OUTPUT_FILE = "fed_dropout_se_attention_results.txt"
    
    # Dataset paths (modify these to your actual paths)
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
    
    print("\n" + "="*80)
    print("🚀 FEDERATED LEARNING WITH SE-ATTENTION FIXED DROPOUT")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Global Epochs: {GLOBAL_EPOCHS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print(f"  - SE Dropout Rate: {SE_DROPOUT_RATE}")
    print(f"  - Aggregation: Mask-Aware FedAvg")
    print("="*80 + "\n")
    
    # ========== INITIALIZE CLIENTS ==========
    client_data = {}
    client_models = {}
    client_test_data = {}
    client_train_samples = {}
    
    print("📚 Loading Client Datasets...")
    
    for client_id, paths in CLIENT_DATASETS.items():
        print(f"\n🔹 Client: {client_id.upper()}")
        
        # Load Training Data
        train_loader, num_classes, class_names = load_client_data(
            paths['train'], train_transform, BATCH_SIZE, is_train=True
        )
        
        if train_loader is None:
            print(f"   ⚠️  Skipping client {client_id} due to data loading error.")
            continue
        
        client_data[client_id] = {
            'train_loader': train_loader,
            'num_classes': num_classes,
            'classes': class_names
        }
        client_train_samples[client_id] = len(train_loader.dataset)
        
        # Load Test Data
        test_loader, _, _ = load_client_data(
            paths['test'], test_transform, BATCH_SIZE, is_train=False
        )
        client_test_data[client_id] = test_loader
        
        # Initialize Client Model
        model = DermaNetAttentionClient(num_classes=num_classes, pretrained=True).to(DEVICE)
        client_models[client_id] = model
    
    if not client_models:
        print("❌ No client models could be initialized. Aborting.")
        return
    
    print(f"\n✅ Successfully loaded {len(client_models)} clients: {', '.join(client_models.keys()).upper()}")
    
    # ========== INITIALIZE MASK MANAGER ==========
    print(f"\n🎭 Initializing SE-Attention Dropout Mask Manager...")
    mask_manager = SEDropoutMaskManager(dropout_rate=SE_DROPOUT_RATE)
    
    # Generate fixed masks for each client
    for client_id, model in client_models.items():
        print(f"\n  🔸 Generating fixed masks for Client {client_id.upper()}:")
        masks = mask_manager.generate_client_mask(client_id, model)
        # Apply masks to initialize the model with dropped weights as zero
        mask_manager.apply_masks_to_model(model, client_id)
    
    # Display coverage statistics
    print(f"\n  📊 SE Parameter Coverage Analysis:")
    coverage_stats = mask_manager.get_coverage_stats()
    for param_name, stats in coverage_stats.items():
        print(f"    {param_name}:")
        print(f"      Mean coverage: {stats['mean_coverage']:.2f} clients")
        print(f"      Zero coverage: {stats['zero_coverage_count']}/{stats['total_params']} params")
    
    # ========== INITIALIZE GLOBAL MODEL ==========
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)
    
    # ========== FEDERATED TRAINING LOOP ==========
    fl_history = defaultdict(lambda: defaultdict(list))
    
    print(f"\n\n🧠 Starting Federated Training for {GLOBAL_EPOCHS} Global Epochs...")
    print("="*80 + "\n")
    
    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\n{'='*20} Global Epoch {g_epoch}/{GLOBAL_EPOCHS} {'='*20}")
        
        # 1. Distribute Global Model and Train Locally
        client_updates = {}
        
        for client_id, model in client_models.items():
            print(f"\n  🔹 Client {client_id.upper()}: Local Training...")
            
            # Set global weights to client's feature extractor
            set_shared_weights(model, global_weights)
            
            # Reapply client's fixed masks (important after receiving global model)
            mask_manager.apply_masks_to_model(model, client_id)
            
            # Local training with fixed masks
            client_shared_weights, local_loss, local_acc = train_client(
                model=model,
                train_loader=client_data[client_id]['train_loader'],
                criterion=FocalLoss(alpha=1, gamma=2).to(DEVICE),
                local_epochs=LOCAL_EPOCHS,
                device=DEVICE,
                mask_manager=mask_manager,
                client_id=client_id
            )
            
            client_updates[client_id] = client_shared_weights
            
            # Record metrics (local, per global epoch)
            fl_history[client_id]['local_loss'].append(local_loss)
            fl_history[client_id]['local_acc'].append(local_acc)
            
            print(f"     ✓ Loss: {local_loss:.4f} | Acc: {local_acc:.4f}")
        
        # 2. Server Aggregation with Mask-Aware FedAvg
        print(f"\n  🔄 Server: Performing Mask-Aware Aggregation...")
        global_weights = fed_avg_with_masks(client_updates, client_train_samples, mask_manager)
        
        # 3. Record global metrics (per global epoch)
        avg_loss = np.mean([fl_history[c]['local_loss'][-1] for c in client_data.keys()])
        avg_acc  = np.mean([fl_history[c]['local_acc'][-1]  for c in client_data.keys()])
        
        fl_history['server']['global_loss'].append(avg_loss)
        fl_history['server']['global_acc'].append(avg_acc)
        
        print(f"  ✓ Global Update Complete | Avg Local Loss: {avg_loss:.4f} | Avg Local Acc: {avg_acc:.4f}")
    
    # ========== FINAL EVALUATION ==========
    print(f"\n\n{'='*80}")
    print("🔍 FINAL EVALUATION ON TEST SETS")
    print("="*80)
    
    final_metrics = {}
    
    for client_id, model in client_models.items():
        print(f"\n📊 Evaluating Client {client_id.upper()} Test Set...")
        
        # Apply final global weights
        set_shared_weights(model, global_weights)
        
        # Apply client's masks for inference
        mask_manager.apply_masks_to_model(model, client_id)
        
        # Evaluate
        test_loader = client_test_data[client_id]
        if test_loader is None:
            continue
        
        acc, f1, loss, true_labels, predictions = evaluate_model(model, test_loader, DEVICE)
        
        final_metrics[client_id] = {
            'acc': acc,
            'f1': f1,
            'loss': loss,
            'classes': client_data[client_id]['classes'],
            'true_labels': true_labels,
            'predictions': predictions
        }
        
        print(f"  ✓ Test Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # ========== SAVE RESULTS ==========
    print(f"\n\n💾 Saving Results to {OUTPUT_FILE}...")
    
    # IMPORTANT: use UTF-8 encoding to avoid UnicodeEncodeError on Windows
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" FEDERATED LEARNING RESULTS: SE-Attention Fixed Dropout (Mask-Aware FedAvg)\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  SE Dropout Rate: {SE_DROPOUT_RATE}\n")
        f.write(f"  Clients: {', '.join(client_data.keys()).upper()}\n")
        f.write(f"  Aggregation: Mask-Aware Weighted Averaging\n")
        f.write(f"  Fixed Masks: Applied to SE-Attention layers only\n")
        f.write("-"*80 + "\n\n")
        
        # SE Coverage Statistics
        f.write("SE-Attention Parameter Coverage:\n")
        f.write("-"*80 + "\n")
        for param_name, stats in coverage_stats.items():
            f.write(f"  {param_name}:\n")
            f.write(f"    Mean coverage: {stats['mean_coverage']:.2f} clients\n")
            f.write(f"    Min coverage: {stats['min_coverage']:.2f}\n")
            f.write(f"    Max coverage: {stats['max_coverage']:.2f}\n")
            f.write(f"    Zero coverage: {stats['zero_coverage_count']}/{stats['total_params']} parameters\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Training History
        f.write("Training History (Per Global Epoch):\n")
        f.write("-"*80 + "\n")
        
        # Header: epoch, global loss/acc, then each client's local loss/acc
        header = f"| {'Epoch':<8} | {'GlobalLoss':<10} | {'GlobalAcc':<10} |"
        for client_id in client_data.keys():
            header += f" {client_id.upper()} Loss | {client_id.upper()} Acc |"
        f.write(header + "\n")
        f.write("|" + "-" * (len(header) - 2) + "|\n")
        
        for g_epoch in range(GLOBAL_EPOCHS):
            line = (
                f"| {g_epoch+1:<8} | "
                f"{fl_history['server']['global_loss'][g_epoch]:<10.4f} | "
                f"{fl_history['server']['global_acc'][g_epoch]:<10.4f} |"
            )
            for client_id in client_data.keys():
                line += f" {fl_history[client_id]['local_loss'][g_epoch]:<10.4f} |"
                line += f" {fl_history[client_id]['local_acc'][g_epoch]:<9.4f} |"
            f.write(line + "\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Final Test Results
        f.write("Final Test Results:\n")
        f.write("-"*80 + "\n")
        
        for client_id, metrics in final_metrics.items():
            f.write(f"\n{'~'*40}\n")
            f.write(f"Client: {client_id.upper()}\n")
            f.write(f"{'~'*40}\n")
            f.write(f"  Test Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Test Accuracy: {metrics['acc']:.4f}\n")
            f.write(f"  Test F1-Score: {metrics['f1']:.4f}\n\n")
            
            # Classification Report
            report = classification_report(
                metrics['true_labels'], 
                metrics['predictions'],
                target_names=metrics['classes'],
                zero_division=0
            )
            f.write("  Classification Report:\n")
            f.write(report + "\n")
            
            # Confusion Matrix
            cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
            f.write("\n  Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            
            # Save confusion matrix plot
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=metrics['classes'],
                           yticklabels=metrics['classes'])
                plt.title(f"Client {client_id.upper()} - Confusion Matrix (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                cm_path = f"confusion_matrix_{client_id}_se_dropout.png"
                plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                f.write(f"  → Confusion matrix saved: {cm_path}\n")
            except Exception as e:
                f.write(f"  → Could not save confusion matrix: {e}\n")
    
    print(f"\n✅ Federated Learning Complete!")
    print(f"   Results saved to: {OUTPUT_FILE}")
    print(f"   Confusion matrices saved as PNG files")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run federated learning
    run_federated_learning()
