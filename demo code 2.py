"""
Federated DermaNet with SE-Attention Fixed Dropout & Mask-Aware Aggregation
============================================================================

Implementation of the Demo_Idea.pdf strategy:
- Fixed dropout masks ONLY in SE-Attention layers (safe for pretrained backbone)
- Mask-aware FedAvg aggregation (avoids zero dilution)
- Client-specific attention specialization for non-IID skin cancer data

Key Innovation:
Instead of applying dropout globally, we surgically apply fixed masks to the
SE-Attention mechanism, allowing each client to specialize in different feature
importance patterns while preserving the pretrained backbone.
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

# Paths - UPDATE THESE TO YOUR DATA
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

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LOCAL_EPOCHS = 4
GLOBAL_EPOCHS = 5
SE_DROPOUT_RATE = 0.3  # Dropout rate for SE-Attention layers only
OUTPUT_FILE = "se_fixed_dropout_results.txt"

# ==============================================================================
# 1. SE-ATTENTION BLOCK WITH FIXED DROPOUT (from Demo_Idea.pdf)
# ==============================================================================

class SEBlockWithFixedDropout(nn.Module):
    """
    Squeeze-Excitation Block with Fixed Client-Specific Dropout Masks.
    
    As per Demo_Idea.pdf recommendation:
    - Apply fixed dropout ONLY to SE layers (not entire network)
    - Each client gets a deterministic mask based on their seed
    - Mask stays fixed throughout training for attention specialization
    """
    
    def __init__(self, channel, reduction=16, dropout_rate=0.3, client_seed=None):
        super(SEBlockWithFixedDropout, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # SE layers
        self.squeeze = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.excite = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Fixed mask for this client (applied to reduced dimension)
        self.dropout_rate = dropout_rate
        self.reduced_dim = channel // reduction
        
        if client_seed is not None:
            # Generate deterministic mask from client seed
            torch.manual_seed(client_seed)
            # Binary mask: 1 = keep, 0 = drop
            mask = torch.bernoulli(torch.full((self.reduced_dim,), 1 - dropout_rate))
            
            # Ensure at least one neuron is active (safety)
            if mask.sum() == 0:
                mask[0] = 1.0
            
            # Register as buffer (moves with model but not trained)
            self.register_buffer('fixed_mask', mask.view(1, -1))
            print(
                f"      SE mask created: {mask.sum().item()}/{self.reduced_dim} active "
                f"({100 * mask.sum().item() / self.reduced_dim:.1f}%)"
            )
        else:
            self.register_buffer('fixed_mask', None)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        
        # Squeeze (channel reduction)
        y = self.squeeze(y)
        y = self.relu(y)
        
        # Apply fixed dropout mask (Demo_Idea.pdf: Step 2)
        if self.fixed_mask is not None:
            y = y * self.fixed_mask  # Zero out dropped channels
        
        # Excite (channel expansion)
        y = self.excite(y)
        y = self.sigmoid(y)
        
        # Scale features by attention weights
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    def get_mask(self):
        """Return the fixed mask for mask-aware aggregation."""
        if self.fixed_mask is not None:
            return self.fixed_mask.cpu().squeeze()
        return None


# ==============================================================================
# 2. FEATURE EXTRACTOR WITH SE-ATTENTION
# ==============================================================================

class DermaNetFeatureExtractor(nn.Module):
    """
    Feature extractor with pretrained EfficientNet backbone + SE-Attention.
    
    Key Design (from Demo_Idea.pdf):
    - Backbone: Preserved with pretrained weights (NOT touched by dropout)
    - SE-Attention: Modified with fixed client-specific dropout
    - This surgical approach avoids destroying pretrained features
    """
    
    def __init__(self, pretrained=True, se_dropout_rate=0.3, client_seed=None):
        super(DermaNetFeatureExtractor, self).__init__()
        
        # Load pretrained EfficientNetB3 backbone (PRESERVED, NOT modified)
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            self.backbone = efficientnet_b3(weights=weights)
        else:
            self.backbone = efficientnet_b3(weights=None)
        
        # Get feature dimension
        self.in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add SE-Attention with fixed dropout (Demo_Idea.pdf strategy)
        self.se_block = SEBlockWithFixedDropout(
            channel=self.in_features,
            reduction=16,
            dropout_rate=se_dropout_rate,
            client_seed=client_seed
        )
    
    def forward(self, x):
        # Extract features from backbone (pretrained, preserved)
        features = self.backbone(x)
        
        # Reshape for SE-Attention
        features = features.unsqueeze(-1).unsqueeze(-1)
        
        # Apply SE-Attention with fixed dropout
        features = self.se_block(features)
        
        # Flatten for classifier
        features = features.view(features.size(0), -1)
        return features


# ==============================================================================
# 3. CLASSIFICATION HEAD (Client-Specific, NOT Aggregated)
# ==============================================================================

class ClassificationHead(nn.Module):
    """Classification head - stays local to each client."""
    
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


# ==============================================================================
# 4. COMPLETE MODEL
# ==============================================================================

class DermaNetAttentionClient(nn.Module):
    """Complete model combining shared extractor + local classifier."""
    
    def __init__(self, num_classes, pretrained=True, se_dropout_rate=0.3, client_seed=None):
        super(DermaNetAttentionClient, self).__init__()
        self.feature_extractor = DermaNetFeatureExtractor(
            pretrained=pretrained,
            se_dropout_rate=se_dropout_rate,
            client_seed=client_seed
        )
        self.classifier_head = ClassificationHead(
            self.feature_extractor.in_features,
            num_classes
        )
        self.num_classes = num_classes
        self.client_seed = client_seed

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier_head(features)
        return output
    
    def get_se_mask(self):
        """Get SE-Attention mask for mask-aware aggregation."""
        return self.feature_extractor.se_block.get_mask()


# ==============================================================================
# 5. FOCAL LOSS (For Class Imbalance)
# ==============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in skin cancer datasets."""
    
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
# 6. MASK-AWARE FEDERATED AVERAGING (from Demo_Idea.pdf)
# ==============================================================================

def mask_aware_fed_avg(client_weights, client_sizes, client_masks):
    """
    Mask-Aware Federated Averaging (Demo_Idea.pdf: Step 2)
    
    Key Innovation:
    - For SE-Attention parameters: only average weights from clients where mask = 1
    - For other parameters: standard weighted averaging
    - This prevents "zero dilution" problem
    
    Args:
        client_weights: Dict of {client_id: state_dict}
        client_sizes: Dict of {client_id: num_samples}
        client_masks: Dict of {client_id: SE mask tensor}
    
    Returns:
        Aggregated global weights
    """
    if not client_weights:
        return None
    
    total_samples = sum(client_sizes.values())
    first_client = list(client_weights.keys())[0]
    global_weights = {}
    
    print("  🔄 Performing mask-aware aggregation...")
    
    for param_name in client_weights[first_client].keys():
        # Identify SE-Attention parameters
        is_se_squeeze = 'se_block.squeeze.weight' in param_name
        is_se_excite = 'se_block.excite.weight' in param_name
        
        if is_se_squeeze or is_se_excite:
            # MASK-AWARE AGGREGATION for SE layers
            param_shape = client_weights[first_client][param_name].shape
            
            if is_se_squeeze:
                # Squeeze layer: apply mask to output dimension (reduced_dim)
                aggregated = torch.zeros(param_shape, dtype=torch.float)
                count = torch.zeros(param_shape[0], dtype=torch.float)  # Per output neuron
                
                for client_id, weights in client_weights.items():
                    client_weight = client_sizes[client_id] / total_samples
                    mask = client_masks[client_id]  # Shape: (reduced_dim,)
                    
                    # Add weighted contribution for active neurons only
                    for i in range(param_shape[0]):  # For each output neuron
                        if i < len(mask) and mask[i] == 1:  # Active in this client
                            aggregated[i] += weights[param_name][i].float() * client_weight
                            count[i] += client_weight
                
                # Normalize by sum of weights for each neuron
                for i in range(param_shape[0]):
                    if count[i] > 0:
                        aggregated[i] /= count[i]
                
                global_weights[param_name] = aggregated
                
            elif is_se_excite:
                # Excite layer: apply mask to input dimension (reduced_dim)
                aggregated = torch.zeros(param_shape, dtype=torch.float)
                count = torch.zeros(param_shape[1], dtype=torch.float)  # Per input neuron
                
                for client_id, weights in client_weights.items():
                    client_weight = client_sizes[client_id] / total_samples
                    mask = client_masks[client_id]
                    
                    # Add weighted contribution for active neurons only
                    for j in range(param_shape[1]):  # For each input neuron
                        if j < len(mask) and mask[j] == 1:
                            aggregated[:, j] += weights[param_name][:, j].float() * client_weight
                            count[j] += client_weight
                
                # Normalize
                for j in range(param_shape[1]):
                    if count[j] > 0:
                        aggregated[:, j] /= count[j]
                
                global_weights[param_name] = aggregated
        
        else:
            # STANDARD AGGREGATION for all other parameters
            aggregated = torch.zeros_like(
                client_weights[first_client][param_name],
                dtype=torch.float
            )
            
            for client_id, weights in client_weights.items():
                client_weight = client_sizes[client_id] / total_samples
                aggregated += weights[param_name].float() * client_weight
            
            global_weights[param_name] = aggregated
    
    print("  ✓ Mask-aware aggregation complete")
    return global_weights


# ==============================================================================
# 7. HELPER FUNCTIONS
# ==============================================================================

def load_client_data(data_path, transform, batch_size=32, is_train=True):
    """Load dataset and return DataLoader."""
    try:
        dataset = ImageFolder(data_path, transform=transform)
        num_classes = len(dataset.classes)
        
        if is_train:
            # Weighted sampling for class imbalance
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
        print(f"   !!! Error loading {data_path}: {e}")
        return None, 0, []


def get_shared_weights(model):
    """Extract shared feature extractor weights."""
    return {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if k.startswith('feature_extractor')
    }


def set_shared_weights(model, shared_weights):
    """Update feature extractor with global weights."""
    model_state = model.state_dict()
    updated_state = {
        k: shared_weights[k].to(model_state[k].device)
        if k.startswith('feature_extractor') else model_state[k]
        for k in model_state
    }
    model.load_state_dict(updated_state)


def train_client(model, train_loader, criterion, local_epochs, device):
    """Train client locally."""
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
    """Evaluate model on test set."""
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
# 8. MAIN FEDERATED LEARNING PIPELINE
# ==============================================================================

def run_federated_learning():
    """Main FL pipeline with SE-Attention fixed dropout."""
    
    print(f"\n{'='*80}")
    print("🚀 FEDERATED LEARNING: SE-ATTENTION FIXED DROPOUT")
    print("   Strategy from Demo_Idea.pdf")
    print(f"{'='*80}\n")
    print(f"Device: {DEVICE}")
    print(f"Global Epochs: {GLOBAL_EPOCHS}")
    print(f"Local Epochs: {LOCAL_EPOCHS}")
    print(f"SE Dropout Rate: {SE_DROPOUT_RATE}")
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
    client_seeds = {}  # Deterministic seeds for each client
    
    print("📚 Loading Client Datasets...\n")
    
    for idx, (client_id, paths) in enumerate(CLIENT_DATASETS.items()):
        print(f"🔹 Client: {client_id.upper()}")
        
        # Generate deterministic seed for this client
        client_seed = abs(hash(client_id)) % (2**32)
        client_seeds[client_id] = client_seed
        print(f"   Client seed: {client_seed}")
        
        # Load training data
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
        
        # Load test data
        test_loader, _, _ = load_client_data(
            paths['test'], test_transform, BATCH_SIZE, is_train=False
        )
        client_test_data[client_id] = test_loader
        
        # Initialize model with client-specific SE dropout
        model = DermaNetAttentionClient(
            num_classes=num_classes,
            pretrained=True,
            se_dropout_rate=SE_DROPOUT_RATE,
            client_seed=client_seed
        ).to(DEVICE)
        client_models[client_id] = model
        print()
    
    if not client_models:
        print("❌ No clients loaded. Aborting.")
        return
    
    print(f"✅ Loaded {len(client_models)} clients: {', '.join(client_models.keys()).upper()}\n")
    
    # Extract SE masks for mask-aware aggregation
    client_masks = {}
    print("🎭 SE-Attention Mask Coverage Analysis:")
    for client_id, model in client_models.items():
        mask = model.get_se_mask()
        if mask is not None:
            client_masks[client_id] = mask
            active = mask.sum().item()
            total = mask.numel()
            print(f"  {client_id.upper()}: {active}/{total} neurons active ({100*active/total:.1f}%)")
    print()
    
    # Initialize global model
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)
    
    # Training history
    fl_history = defaultdict(lambda: defaultdict(list))
    
    print(f"{'='*80}")
    print("🧠 Starting Federated Training")
    print(f"{'='*80}\n")
    
    # Federated training loop
    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\n{'─'*80}")
        print(f"Global Epoch {g_epoch}/{GLOBAL_EPOCHS}")
        print(f"{'─'*80}")
        
        client_updates = {}
        
        # Local training
        for client_id, model in client_models.items():
            print(f"\n  🔹 {client_id.upper()}: Local training...")
            
            # Distribute global weights
            set_shared_weights(model, global_weights)
            
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
        
        # Server aggregation with mask-awareness
        print(f"\n  🔄 Server: Mask-aware aggregation...")
        global_weights = mask_aware_fed_avg(client_updates, client_train_samples, client_masks)
        
        # Record global metrics (average over clients for this round)
        avg_loss = np.mean([fl_history[c]['local_loss'][-1] for c in client_data.keys()])
        avg_acc = np.mean([fl_history[c]['local_acc'][-1] for c in client_data.keys()])
        
        fl_history['server']['global_loss'].append(avg_loss)
        fl_history['server']['global_acc'].append(avg_acc)
        
        print(f"  ✓ Global epoch complete | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")
    
    # Final evaluation
    print(f"\n\n{'='*80}")
    print("🔍 FINAL EVALUATION")
    print(f"{'='*80}\n")
    
    final_metrics = {}
    
    for client_id, model in client_models.items():
        print(f"📊 {client_id.upper()} Test Set:")
        
        # Apply global weights
        set_shared_weights(model, global_weights)
        
        # Evaluate
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
    print(f"💾 Saving results to {OUTPUT_FILE}...\n")
    
    # IMPORTANT: UTF-8 to avoid UnicodeEncodeError on Windows for characters like "→"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(" FEDERATED LEARNING RESULTS: SE-ATTENTION FIXED DROPOUT\n")
        f.write(" Strategy from Demo_Idea.pdf\n")
        f.write("="*80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  SE Dropout Rate: {SE_DROPOUT_RATE}\n")
        f.write(f"  Clients: {', '.join(client_data.keys()).upper()}\n")
        f.write(f"  Aggregation: Mask-Aware FedAvg (SE layers only)\n")
        f.write("-"*80 + "\n\n")
        
        # SE Mask coverage
        f.write("SE-Attention Mask Coverage:\n")
        for client_id, mask in client_masks.items():
            active = mask.sum().item()
            total = mask.numel()
            f.write(f"  {client_id.upper()}: {active}/{total} active ({100*active/total:.1f}%)\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Training history
        f.write("Training History:\n")
        f.write("-"*80 + "\n")
        
        # Header now includes global loss and global acc
        header = f"| {'Epoch':<8} | {'GlobalLoss':<10} | {'GlobalAcc':<10} |"
        for cid in client_data.keys():
            header += f" {cid.upper()} Loss | {cid.upper()} Acc |"
        f.write(header + "\n")
        f.write("|" + "-" * (len(header) - 2) + "|\n")
        
        for ep in range(GLOBAL_EPOCHS):
            line = (
                f"| {ep+1:<8} | "
                f"{fl_history['server']['global_loss'][ep]:<10.4f} | "
                f"{fl_history['server']['global_acc'][ep]:<10.4f} |"
            )
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
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=metrics['classes'],
                    yticklabels=metrics['classes']
                )
                plt.title(f"{client_id.upper()} - Confusion Matrix (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                cm_file = f"confusion_matrix_{client_id}_se_dropout.png"
                plt.savefig(cm_file, dpi=300, bbox_inches='tight')
                plt.close()
                f.write(f"  → Confusion matrix saved: {cm_file}\n\n")
            except Exception as e:
                f.write(f"  → Could not save confusion matrix: {e}\n\n")
            
            f.write("-"*80 + "\n\n")
    
    print(f"✅ Training Complete!")
    print(f"   Results saved: {OUTPUT_FILE}")
    print(f"   Confusion matrices saved as PNG files")
    print(f"\n{'='*80}\n")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run federated learning
    run_federated_learning()
