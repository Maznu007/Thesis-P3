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
# 1. MODEL AND LOSS DEFINITIONS (ADAPTED FROM USER'S FILE)
# ==============================================================================

# Squeeze-Excitation Block
class SEBlock(nn.Module):
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

# Feature Extractor (Shared layers for Averaging)
class DermaNetFeatureExtractor(nn.Module):
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
        
        # Add SE attention
        self.se_block = SEBlock(self.in_features, reduction=16)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        
        # Apply SE attention
        features = self.se_block(features)
        features = features.view(features.size(0), -1)
        return features

# Classification Head (Client-specific layers)
class ClassificationHead(nn.Module):
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

# Full Model (Combines shared and client-specific parts)
class DermaNetAttentionClient(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DermaNetAttentionClient, self).__init__()
        self.feature_extractor = DermaNetFeatureExtractor(pretrained=pretrained)
        self.classifier_head = ClassificationHead(self.feature_extractor.in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier_head(features)
        return output

# Focal Loss (handles class imbalance)
class FocalLoss(nn.Module):
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
# 2. HELPER FUNCTIONS: DATA LOADING AND FL OPERATIONS
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
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print(f"   -> Loaded {len(dataset)} samples with {num_classes} classes.")
        return loader, num_classes, dataset.classes
    except Exception as e:
        # Use a print statement instead of error logging in the code itself, 
        # as this is running in an environment where the path might be the issue.
        print(f"   !!! Error loading data from {data_path}: {e}")
        return None, 0, []

def get_shared_weights(model):
    """Extracts weights for the shared feature extractor layers."""
    return {k: v.cpu() for k, v in model.state_dict().items() if k.startswith('feature_extractor')}

def set_shared_weights(model, shared_weights):
    """Updates the shared feature extractor layers with global weights."""
    model_state = model.state_dict()
    updated_state = {
        k: shared_weights[k].to(model_state[k].device) if k.startswith('feature_extractor') else model_state[k]
        for k in model_state
    }
    model.load_state_dict(updated_state)

def fed_avg(client_weights, client_sizes):
    """
    Performs Federated Averaging on the shared weights.
    Client sizes are used as aggregation weights.
    
    FIXED: Explicitly casts global_weights to float to prevent RuntimeError 
    when accumulating float-scaled weights into int tensors (like BatchNorm stats).
    """
    if not client_weights:
        return None

    # Calculate total samples
    total_samples = sum(client_sizes.values())
    
    # Initialize global weights with the first client's shared weights
    first_client_id = list(client_weights.keys())[0]
    global_weights = deepcopy(client_weights[first_client_id])
    
    # CRITICAL FIX: Ensure all tensors are float type before averaging
    for key in global_weights.keys():
        # Convert to float and initialize to zero for summation
        # This prevents the "Float can't be cast to Long" error
        global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float)

    # Federated Averaging: weighted sum of client weights
    for client_id, weights in client_weights.items():
        weight_factor = client_sizes[client_id] / total_samples
        for key in global_weights.keys():
            # Add scaled client weight to the global sum. 
            # .add_() is an in-place addition operation.
            global_weights[key].add_(weights[key], alpha=weight_factor)
            
    return global_weights


# ==============================================================================
# 3. TRAINING AND EVALUATION LOGIC
# ==============================================================================

def train_client(model, train_loader, criterion, local_epochs, device):
    """Performs local training for a client."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for epoch in range(local_epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Tracking
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return get_shared_weights(model), avg_loss, avg_acc

def evaluate_model(model, data_loader, device):
    """Evaluates a model (local or global) on a dataset."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0
    # Use reduction='sum' for correct average loss calculation
    criterion = FocalLoss(alpha=1, gamma=2, reduction='sum') 
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            # Loss calculation
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)

    acc = accuracy_score(all_labels, all_preds)
    # Calculate F1 score using 'weighted' average because class sizes vary and focal loss suggests imbalance
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_loss = total_loss / total_samples
    
    return acc, f1, avg_loss, np.array(all_labels), np.array(all_preds)


# ==============================================================================
# 4. MAIN FEDERATED LEARNING EXECUTION
# ==============================================================================

def run_federated_learning():
    """Main function to run the FedAvg simulation."""
    
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GLOBAL_EPOCHS = 5
    LOCAL_EPOCHS = 4
    BATCH_SIZE = 32
    OUTPUT_FILE = "federated_results.txt"
    
    # Client Data Paths (as provided by the user)
    client_paths = {
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

    # --- Data Loading and Client Setup ---
    print("🚀 Setting up Federated Learning Clients...")
    client_data = {}
    client_train_samples = {}
    client_models = {}
    client_test_data = {}
    
    # Minimal, clinically-realistic augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Test transform: NO augmentation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for client_id, paths in client_paths.items():
        print(f"📦 Loading data for Client {client_id.upper()}...")
        # Load Training Data
        train_loader, num_classes, classes = load_client_data(
            paths['train'], train_transform, BATCH_SIZE, is_train=True
        )
        if train_loader is None: continue

        client_data[client_id] = {'train_loader': train_loader, 'num_classes': num_classes, 'classes': classes}
        client_train_samples[client_id] = len(train_loader.dataset)
        
        # Load Test Data (for final evaluation)
        test_loader, _, _ = load_client_data(
            paths['test'], test_transform, BATCH_SIZE, is_train=False
        )
        client_test_data[client_id] = test_loader
        
        # Initialize Client Model (with correct number of classes)
        model = DermaNetAttentionClient(num_classes=num_classes, pretrained=True).to(DEVICE)
        client_models[client_id] = model

    # --- Server Initialization ---
    # The server model is conceptually the feature extractor (Global Model),
    # but we represent it by the HAM client model's shared weights initially.
    # Note: We must ensure at least one client loaded successfully.
    if not client_models:
        print("❌ No client models could be initialized. Aborting.")
        return

    # Use one client's feature extractor as the initial global model state
    initial_model = list(client_models.values())[0]
    global_weights = get_shared_weights(initial_model)
    
    # --- Federated Training Loop ---
    fl_history = defaultdict(lambda: defaultdict(list))
    
    print(f"\n🧠 Starting FedAvg Training for {GLOBAL_EPOCHS} Global Epochs...")

    for g_epoch in range(1, GLOBAL_EPOCHS + 1):
        print(f"\n======== Global Epoch {g_epoch}/{GLOBAL_EPOCHS} ========")
        
        # 1. Distribute Global Model and Train Locally
        client_updates = {}
        
        for client_id, model in client_models.items():
            print(f"  -> Client {client_id.upper()}: Local Training...")
            
            # Set global weights to client's feature extractor
            set_shared_weights(model, global_weights)
            
            # Local training
            client_shared_weights, local_loss, local_acc = train_client(
                model=model, 
                train_loader=client_data[client_id]['train_loader'], 
                criterion=FocalLoss(alpha=1, gamma=2).to(DEVICE), 
                local_epochs=LOCAL_EPOCHS, 
                device=DEVICE
            )
            
            client_updates[client_id] = client_shared_weights
            
            # Record local metrics
            fl_history[client_id]['local_loss'].append(local_loss)
            fl_history[client_id]['local_acc'].append(local_acc)
            print(f"     [Client {client_id.upper()}] Loss: {local_loss:.4f} | Acc: {local_acc:.4f}")
            
        
        # 2. Server Aggregation (FedAvg)
        print("  -> Server: Aggregating Weights (FedAvg)...")
        global_weights = fed_avg(client_updates, client_train_samples)
        
        # 3. Global Model Evaluation on Training Data (for reporting global convergence)
        # For simplicity and due to varying heads, we will check the aggregate local loss/acc
        avg_loss = np.mean([fl_history[c]['local_loss'][-1] for c in client_data.keys()])
        
        fl_history['server']['global_loss'].append(avg_loss)
        
        print(f"  -> Global Model Update Complete. Avg Local Loss: {avg_loss:.4f}")
    
    # --- 5. Final Global Model Evaluation on Test Sets ---
    print(f"\n\n===========================================")
    print("🔍 Final Global Model Evaluation on Test Sets")
    print("===========================================")
    
    final_metrics = {}
    
    for client_id, model in client_models.items():
        print(f"\nEvaluating on Client {client_id.upper()} Test Set...")
        
        # Apply the final global weights to the client's feature extractor
        set_shared_weights(model, global_weights)
        
        # Evaluate
        test_loader = client_test_data[client_id]
        if test_loader is None: continue
        
        acc, f1, loss, true_labels, predictions = evaluate_model(model, test_loader, DEVICE)
        
        final_metrics[client_id] = {
            'acc': acc,
            'f1': f1,
            'loss': loss,
            'classes': client_data[client_id]['classes'],
            'true_labels': true_labels,
            'predictions': predictions
        }
        
        print(f"  -> Test Loss: {loss:.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

    
    # --- 6. Save All Results to Text File ---
    print(f"\n\n💾 Saving all results to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(" FEDERATED LEARNING RESULTS: DermaNet-Attention (FedAvg) \n")
        f.write("=" * 80 + "\n")
        f.write(f"Parameters:\n")
        f.write(f"  Global Epochs: {GLOBAL_EPOCHS}\n")
        f.write(f"  Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"  Clients: {', '.join(client_data.keys()).upper()}\n")
        f.write(f"  Aggregation: Weighted Averaging (by sample size)\n")
        f.write(f"  Averaged Layers: Feature Extractor (Backbone + SE-Attention)\n")
        f.write("-" * 80 + "\n\n")

        
        # A. Per-Global Epoch Training History
        f.write("A. Per-Global Epoch Training History\n")
        f.write("-" * 80 + "\n")
        
        header = f"| {'G. Epoch':<10} | {'Avg. Local Loss':<15} |"
        for client_id in client_data.keys():
            header += f" {client_id.upper()} Loss | {client_id.upper()} Acc |"
        f.write(header + "\n")
        
        f.write("|" + "---" * 27 + "\n") # Separator line
        
        for g_epoch in range(GLOBAL_EPOCHS):
            line = f"| {g_epoch + 1:<10} | {fl_history['server']['global_loss'][g_epoch]:<15.4f} |"
            for client_id in client_data.keys():
                line += f" {fl_history[client_id]['local_loss'][g_epoch]:<10.4f} | {fl_history[client_id]['local_acc'][g_epoch]:<9.4f} |"
            f.write(line + "\n")
            
        f.write("\n" + "=" * 80 + "\n\n")
        
        
        # B. Final Test Results on Client Test Sets
        f.write("B. Final Global Model Test Results\n")
        f.write("-" * 80 + "\n")
        
        for client_id, metrics in final_metrics.items():
            f.write(f"\n--- Client: {client_id.upper()} ---\n")
            f.write(f"  Test Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Test Accuracy: {metrics['acc']:.4f}\n")
            f.write(f"  Test F1-Score (Weighted): {metrics['f1']:.4f}\n")
            
            # Classification Report
            report = classification_report(
                metrics['true_labels'], metrics['predictions'], 
                target_names=metrics['classes'], zero_division=0
            )
            f.write("\n  Classification Report:\n")
            f.write(report)
            
            # Confusion Matrix (optional, but good to include raw data)
            cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
            f.write("\n  Confusion Matrix (rows: True, cols: Predicted):\n")
            f.write(str(cm) + "\n")
            
            # Generate and save Confusion Matrix Plot
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=metrics['classes'], yticklabels=metrics['classes'])
                plt.title(f"Client {client_id.upper()} Confusion Matrix (F1={metrics['f1']:.4f})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                # Save to a dedicated file in the directory
                conf_matrix_path = f"conf_matrix_{client_id}_final.png"
                plt.savefig(conf_matrix_path, dpi=300)
                plt.close()
                f.write(f"  -> Confusion Matrix Plot saved to: {conf_matrix_path}\n")
            except Exception as e:
                f.write(f"  -> Could not generate confusion matrix plot: {e}\n")


    print(f"\n✅ Federated training complete!")
    print(f"   Final model evaluated on all test sets.")
    print(f"   All results saved to {OUTPUT_FILE} and confusion matrix plots generated.")

if __name__ == "__main__":
    # Suppress warnings if needed
    # import warnings
    # warnings.filterwarnings("ignore")
    run_federated_learning()