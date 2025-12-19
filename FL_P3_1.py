import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import regnet_y_32gf, RegNet_Y_32GF_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================================================================
# 1. CONSTANTS AND CONFIGURATION (!!! ADJUST THESE !!!)
# =============================================================================

# --- Input Paths ---
BASE_DIRS = {
    'ham': r"D:\dataset\ham\organized_ham",
    'isic': r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration",
    'pad': r"D:\dataset\pad\organized_pad"
}
# !!! IMPORTANT: Single directory containing ALL test images (from all 3 datasets)
COMBINED_TEST_DIR = r"D:\dataset\combined_test_data\test" 

# --- Output Path ---
BASE_SAVE_DIR = r"D:\dataset\p3" 
os.makedirs(BASE_SAVE_DIR, exist_ok=True) 

# --- FL Parameters ---
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 
NUM_CLASSES = 7 # !!! ADJUST THIS to the largest number of classes across your datasets !!!

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- File Names ---
RESULTS_FILE = os.path.join(BASE_SAVE_DIR, "federated_result.txt")
CONF_MATRIX_FILE = os.path.join(BASE_SAVE_DIR, "confusion_matrix_final.png")
FINAL_MODEL_FILE = os.path.join(BASE_SAVE_DIR, "federated_regnet_model_final.pth")

# =============================================================================
# 2. MODEL AND DATA UTILITIES
# =============================================================================

def get_regnet_model(num_classes):
    """Initializes RegNetY-32GF with a custom final layer."""
    # print("🔍 Running: get_regnet_model - Initializing model architecture.")
    model = regnet_y_32gf(weights=RegNet_Y_32GF_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.fc.requires_grad_(True) 
    return model.to(DEVICE)

def load_data_client(base_dir):
    """Loads the training DataLoader for a single client."""
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found for client {os.path.basename(base_dir)}: {train_dir}")
    train_dataset = ImageFolder(train_dir, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # print(f"  ➡️ Running: load_data_client - Loaded {len(train_dataset)} samples.")
    return train_loader, len(train_dataset)

def load_combined_test_data(combined_test_dir):
    """Loads the centralized test DataLoader."""
    if not os.path.exists(combined_test_dir):
        raise FileNotFoundError(f"Combined test data directory not found: {combined_test_dir}")
    test_dataset = ImageFolder(combined_test_dir, transform=TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"  ✅ Running: load_combined_test_data - Loaded {len(test_dataset)} test samples.")
    return test_loader, test_dataset.classes

def fed_avg(global_model_state, client_states, client_sizes):
    """Performs Weighted Federated Averaging (FedAvg)."""
    # print("  ⚖️ Running: fed_avg - Calculating weighted average.")
    total_size = sum(client_sizes)
    weights = [size / total_size for size in client_sizes]
    averaged_state = copy.deepcopy(global_model_state)
    for key in averaged_state.keys():
        averaged_state[key] = torch.zeros_like(averaged_state[key])
    for weight, state in zip(weights, client_states):
        for key in averaged_state.keys():
            averaged_state[key] += weight * state[key]
    # print("  ✅ Running: fed_avg - Aggregation complete.")
    return averaged_state

# =============================================================================
# 3. CLIENT TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def client_update(client_id, global_weights, train_loader):
    """Performs local training and tracks per-epoch metrics."""
    print(f"  ⚙️ Running: client_update for {client_id} - Loading global weights.")
    local_model = get_regnet_model(NUM_CLASSES)
    local_model.load_state_dict(global_weights)
    local_model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
    
    local_history = []
    
    # 1. Local Training Loop
    for epoch in range(LOCAL_EPOCHS):
        running_loss = 0.0
        corrects = 0
        total = 0
        
        # print(f"  🔄 Running: Local Epoch {epoch+1}/{LOCAL_EPOCHS}...")
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Per-Local-Epoch Metrics (Tracking)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += images.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
        
        local_history.append((epoch_loss, epoch_acc.item()))
    
    print(f"  ⬆️ Running: client_update for {client_id} - Training finished, returning weights.")
    return local_model.state_dict(), local_history

def server_evaluate(model, test_loader, class_names):
    """Evaluates the global model on the centralized test set."""
    print("  ⭐ Running: server_evaluate - Starting global model test.")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += images.size(0)

    # Per-Global-Round Metrics
    test_loss = running_loss / total
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Final Metrics (Classification Report and Confusion Matrix objects)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("  📊 Running: server_evaluate - Evaluation complete.")
    return test_loss, test_acc, test_f1, report, cm, class_names

# =============================================================================
# 4. SERVER FL LOOP (MAIN EXECUTION)
# =============================================================================

def federated_training_loop():
    print(f"🚀 Starting Federated Training. Results will be saved to: {BASE_SAVE_DIR}")
    
    # --- Data and Client Setup ---
    print("1️⃣ Setting up clients and data loaders...")
    client_data = {
        'ham': load_data_client(BASE_DIRS['ham']),
        'isic': load_data_client(BASE_DIRS['isic']),
        'pad': load_data_client(BASE_DIRS['pad'])
    }
    client_names = list(client_data.keys())
    client_loaders = [data[0] for data in client_data.values()]
    client_sizes = [data[1] for data in client_data.values()]
    
    test_loader, class_names = load_combined_test_data(COMBINED_TEST_DIR)

    # --- Initialization ---
    print("2️⃣ Initializing Global Model...")
    global_model = get_regnet_model(NUM_CLASSES)
    global_weights = global_model.state_dict()
    
    # --- Initial Logging Setup ---
    with open(RESULTS_FILE, "w") as f:
        f.write(f"======================================================\n")
        f.write(f"FEDERATED LEARNING EXPERIMENT RESULTS\n")
        f.write(f"MODEL: RegNetY-32GF (Clients: {', '.join(client_names)})\n")
        f.write(f"GLOBAL ROUNDS: {GLOBAL_ROUNDS} | LOCAL EPOCHS: {LOCAL_EPOCHS} | NUM_CLASSES: {NUM_CLASSES}\n")
        f.write(f"------------------------------------------------------\n")
        f.write(f"Data Sizes (for FedAvg weighting): {client_sizes}\n")
        f.write(f"======================================================\n")

    # --- Global FL Loop ---
    final_report = ""
    final_cm = None
    final_acc = 0.0
    
    for t in range(GLOBAL_ROUNDS):
        round_num = t + 1
        print(f"\n==================== GLOBAL ROUND {round_num}/{GLOBAL_ROUNDS} ====================")
        
        local_weights_list = []
        local_history_list = {}
        
        # 1. Client Training and Upload
        for client_name, loader in zip(client_names, client_loaders):
            print(f"3️⃣ Starting: Client {client_name} training...")
            updated_weights, local_history = client_update(client_name, global_weights, loader)
            local_weights_list.append(updated_weights)
            local_history_list[client_name] = local_history
            
        # 2. Server Aggregation (FedAvg)
        print("4️⃣ Server: Aggregating client updates (FedAvg)...")
        global_weights = fed_avg(global_weights, local_weights_list, client_sizes)
        global_model.load_state_dict(global_weights)
        
        # 3. Server Evaluation (Per-Global-Round Metrics)
        print("5️⃣ Server: Evaluating global model on combined test set...")
        global_loss, global_acc, global_f1, report, cm, _ = server_evaluate(
            global_model, test_loader, class_names
        )
        final_report, final_cm, final_acc = report, cm, global_acc # Keep the latest metrics
        
        # 4. Log Round Results
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\n--- ROUND {round_num}/{GLOBAL_ROUNDS} ---\n")
            
            # Log Per-Local-Epoch Metrics
            for client_name, history in local_history_list.items():
                f.write(f"[CLIENT: {client_name}]\n")
                for epoch, (loss, acc) in enumerate(history):
                    f.write(f"  Local Epoch {epoch+1}: Train Loss={loss:.4f}, Train Acc={acc:.4f}\n")
                    
            # Log Per-Global-Round Metrics
            f.write("GLOBAL MODEL EVALUATION:\n")
            f.write(f"  Test Loss: {global_loss:.4f}\n")
            f.write(f"  Test Accuracy: {global_acc:.4f}\n")
            f.write(f"  Test F1-Score (Weighted): {global_f1:.4f}\n")
        
        print(f"6️⃣ Global Round {round_num} Complete: Acc={global_acc:.4f}, Loss={global_loss:.4f}")

    # --- Final Output ---
    print(f"\n==================== TRAINING COMPLETE ====================")
    
    # 7. Final Metrics (Classification Report)
    print("7️⃣ Saving Final Metrics to result.txt...")
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n\n======================================================\n")
        f.write(f"FINAL GLOBAL MODEL METRICS (Round {GLOBAL_ROUNDS})\n")
        f.write(f"======================================================\n")
        f.write(f"FINAL TEST ACCURACY: {final_acc:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(final_report + "\n")

    # 8. Save Final Model
    print(f"8️⃣ Saving Final Model weights to: {FINAL_MODEL_FILE}...")
    torch.save(global_model.state_dict(), FINAL_MODEL_FILE)
    print(f"💾 Model saved successfully.")
    
    # 9. Save Confusion Matrix
    if final_cm is not None:
        print("9️⃣ Generating and saving Confusion Matrix image...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(final_cm, annot=True, fmt='d',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Federated RegNetY Final Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_FILE)
        plt.close()
        print(f"🖼️ Confusion matrix image saved to: {CONF_MATRIX_FILE}")
    
    print(f"✅ All results saved successfully to {BASE_SAVE_DIR}")

if __name__ == '__main__':
    federated_training_loop()