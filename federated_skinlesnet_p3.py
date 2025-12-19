import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import warnings

# Suppress UserWarning about num_workers for simplicity in this environment
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# CONFIGURATION
# ====================================================================
print("--- [0] Initializing Configuration ---")

# Client Paths
CLIENT_TRAIN_DIRS = {
    'ham': r"D:\dataset\ham\organized_ham\train",
    'isic': r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Train",
    'pad': r"D:\dataset\pad\organized_pad\train",
}
CLIENT_TEST_DIRS = {
    'ham': r"D:\dataset\ham\organized_ham\test",
    'isic': r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration\Test",
    'pad': r"D:\dataset\pad\organized_pad\test",
}
OUTPUT_DIR = r"D:\dataset\fl results new folder named vgg16 federated result p3"
MODEL_NAME = "skinlesnet_fedavg_global_model.pth"
RESULTS_FILE = "fedavg_skinlesnet_results.txt"

# FedAvg Parameters
LOCAL_EPOCHS = 4
GLOBAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224 # SkinLesNet expects 224x224 input

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
print(f"--- Output directory created at: {OUTPUT_DIR} ---")
print(f"--- Results will be saved to: {results_path} ---")

# Global variables for logging
GLOBAL_LOGS = []
CRITERION = nn.CrossEntropyLoss()

# ====================================================================
# MODEL ARCHITECTURE (SkinLesNet)
# ====================================================================

class SkinLesNet(nn.Module):
    def __init__(self, num_classes):
        super(SkinLesNet, self).__init__()
        # Feature extractor definition (Shared across clients)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Output: 32 channels, 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Output: 64 channels, 56x56

            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Output: 64 channels, 28x28

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Output: 128 channels, 14x14

            nn.Dropout(p=0.5),
        )
        # Classifier definition (Final layer is client-specific)
        # Input size for the first linear layer is 128 channels * 14*14
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 64), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes) # This layer's output size must be customized
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_model(num_classes):
    """Initializes SkinLesNet with the correct output size."""
    print(f"--- Initializing SkinLesNet with {num_classes} classes. ---")
    return SkinLesNet(num_classes=num_classes).to(device)

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def save_log(message):
    """Saves a message to the global log and prints it."""
    GLOBAL_LOGS.append(message)
    print(message)

def get_dataloaders(client_name, train_dir, test_dir):
    """Loads dataset and returns loaders, class names, and class count."""
    print(f"--- [1.1] Setting up data for client: {client_name} ---")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Using simple mean/std for custom model
        transforms.Normalize([0.5]*3, [0.5]*3) 
    ])

    train_dataset = ImageFolder(train_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)

    # num_workers=0 to prevent potential issues in notebook environments
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    data_size = len(train_dataset)

    print(f"--- Client {client_name}: Classes={num_classes}, Train Size={data_size} ---")
    return train_loader, test_loader, num_classes, class_names, data_size

def client_update(model, train_loader, local_epochs, criterion, client_name):
    """Performs local training on a client's data."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    local_log = []
    
    save_log(f"--- [2.2] Starting local training for Client {client_name} (Epochs: {local_epochs}) ---")

    for epoch in range(local_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            
            # Print at the start of epoch or every 100 batches
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"--- Client {client_name}, Local Epoch {epoch+1}/{local_epochs}: Batch {batch_idx}/{len(train_loader)} ---")

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            # Fix for 'RuntimeError: result type Float can't be cast to the desired output type Long'
            # outputs are floats, but we get the Long-type indices (preds) for comparison
            _, preds = torch.max(outputs, 1) 
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total
        
        log_line = (f"  > C-{client_name} L-Epoch {epoch+1}/{local_epochs} "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        local_log.append(log_line)
        save_log(log_line)
        
    return model.state_dict(), total, local_log # Return total for aggregation weight

def fed_avg(global_model, client_weights, client_sizes):
    """
    Aggregates client weights using FedAvg. 
    Crucially, it skips aggregation for the final layer ('classifier.3.*')
    where dimensions may differ (non-IID classes).
    """
    save_log("--- [3.1] Starting FedAvg Aggregation ---")
    
    # Calculate total data size
    total_size = sum(client_sizes.values())
    
    # Initialize the averaged weights (same structure as global_model)
    global_weights = global_model.state_dict()
    
    # Iterate over all parameters in the global model
    for name in global_weights.keys():
        
        # 1. Skip the client-specific final classification layer
        if 'classifier.3' in name:
            save_log(f"--- Skipping aggregation for {name} (Client-specific layer) ---")
            continue
            
        # 2. For all shared layers, initialize the averaged tensor
        avg_tensor = torch.zeros_like(global_weights[name])
        
        # 3. Aggregate weights
        for client_name in client_weights.keys():
            client_weight = client_sizes[client_name] / total_size
            local_param = client_weights[client_name].get(name)

            # CRITICAL CHECK: Ensure the local parameter exists and matches the shape of the global tensor
            if local_param is None:
                save_log(f"--- Warning: Parameter {name} missing in client {client_name}. Skipping contribution. ---")
                continue
            
            if local_param.shape != global_weights[name].shape:
                # This should only happen if the layer naming/skipping logic failed, but we check defensively.
                save_log(f"--- ERROR: Shape mismatch for {name} in client {client_name} (Global: {global_weights[name].shape} vs Local: {local_param.shape}). Skipping contribution. ---")
                continue

            # Perform the weighted aggregation
            avg_tensor += client_weight * local_param
            
        # 4. Update the global model with the averaged tensor
        global_weights[name].data.copy_(avg_tensor)
        
    global_model.load_state_dict(global_weights)
    save_log("--- FedAvg Aggregation Complete ---")
    return global_model

def evaluate(model, loader, criterion, device):
    """Evaluates the model on a test set."""
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    avg_acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, avg_acc

def final_predict_report(model, loader, class_names, client_name):
    """Generates the final classification report."""
    save_log(f"--- [4.1] Generating final report for {client_name} Test Set ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Ensure class names are used consistently for reporting
    target_names = [str(c) for c in class_names]

    # Recalculate evaluation metrics as a combined single pass for the test set
    test_loss, test_acc = evaluate(model, loader, CRITERION, device)

    # The classification_report needs to be handled carefully when classes differ
    # We rely on the local model's class size for its predictions
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=False, zero_division=0)

    report_log = (
        f"\n======================================================\n"
        f"FINAL TEST RESULTS: {client_name.upper()} (Classes: {len(class_names)})\n"
        f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n"
        f"======================================================\n"
        f"{report}"
    )
    save_log(report_log)
    return report_log

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def run_fedavg():
    # --- [1] Data Preparation ---
    client_data = {}
    max_classes = 0
    
    print("--- [1.0] Preparing all client datasets and finding max classes ---")
    for client_name, train_dir in CLIENT_TRAIN_DIRS.items():
        if not os.path.exists(train_dir):
             raise FileNotFoundError(f"Train path not found for {client_name}: {train_dir}")
        test_dir = CLIENT_TEST_DIRS[client_name]
        if not os.path.exists(test_dir):
             raise FileNotFoundError(f"Test path not found for {client_name}: {test_dir}")
             
        train_loader, test_loader, num_classes, class_names, data_size = get_dataloaders(
            client_name, train_dir, test_dir
        )
        
        client_data[client_name] = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'num_classes': num_classes,
            'class_names': class_names,
            'data_size': data_size,
        }
        max_classes = max(max_classes, num_classes)
    
    # --- [2] Model Initialization ---
    print(f"--- [2.0] Initializing Global Model (Max Classes: {max_classes}) ---")
    # The global model is initialized with the max number of classes to hold the aggregated feature weights
    global_model = create_model(max_classes) 
    
    # --- [3] Federated Learning Loop ---
    save_log(f"\n--- [3.0] Starting Federated Training (Global Epochs: {GLOBAL_EPOCHS}, Local Epochs: {LOCAL_EPOCHS}) ---\n")
    
    # Store aggregated test results for the global model
    global_test_results = {} 
    
    for global_epoch in range(1, GLOBAL_EPOCHS + 1):
        save_log(f"\n=======================================================================")
        save_log(f"| GLOBAL EPOCH {global_epoch}/{GLOBAL_EPOCHS} |")
        save_log(f"=======================================================================")
        
        client_weights = {}
        client_sizes = {}
        local_logs_per_epoch = []
        
        # Step 1: Clients Train Locally
        for client_name, data in client_data.items():
            
            save_log(f"\n--- [2.1] Preparing Client {client_name} ---")
            
            # Create the client's local model instance with the correct local class count
            local_model = create_model(data['num_classes'])
            
            # Load the current aggregated shared weights from the global model
            current_global_weights = global_model.state_dict()
            
            # Create a dictionary for the weights that match the local model's structure
            # Initialize with all the local weights (including the local classifier.3's random init)
            new_local_state_dict = local_model.state_dict()
            
            # Copy only the shared weights (those NOT containing 'classifier.3')
            # from the global model to the new_local_state_dict.
            # This ensures we skip the layer that has mismatched dimensions.
            for name, param in current_global_weights.items():
                if 'classifier.3' not in name:
                    # Check if the weight exists in the local model (it always should for shared layers)
                    # and ensure the tensor shapes match (they always should for shared layers)
                    if name in new_local_state_dict and param.shape == new_local_state_dict[name].shape:
                        new_local_state_dict[name].copy_(param)
                    else:
                        # This should ideally only print if a shared layer's size somehow changed,
                        # but it's good practice for debugging.
                        save_log(f"--- Warning: Skipping parameter {name} due to shape mismatch or absence. ---")


            # Load the filtered state dictionary into the local model.
            # The client-specific classifier.3 layer remains its initial random state, 
            # which is what we want for local training.
            local_model.load_state_dict(new_local_state_dict)

            # Perform local training
            local_state_dict, total_size_client, local_log = client_update(
                local_model, 
                data['train_loader'], 
                LOCAL_EPOCHS, 
                CRITERION, 
                client_name
            )
            
            # Store the updated state dict and size for aggregation
            client_weights[client_name] = local_state_dict
            client_sizes[client_name] = total_size_client
            local_logs_per_epoch.extend(local_log)

        # Step 2: Server Aggregates Weights
        global_model = fed_avg(global_model, client_weights, client_sizes)
        
        # Step 3: Global Model Evaluation
        global_epoch_metrics = []
        for client_name, data in client_data.items():
            # Evaluation requires a temporary model matching the client's class size.
            
            # 1. Create a temporary model with the client's class size.
            temp_model = create_model(data['num_classes'])
            
            # 2. Copy shared weights from global model to temp model.
            global_state_dict = global_model.state_dict()
            temp_state_dict = temp_model.state_dict()
            
            # Use the same filtering logic for evaluation
            for name, param in global_state_dict.items():
                 if 'classifier.3' not in name:
                    if name in temp_state_dict and param.shape == temp_state_dict[name].shape:
                        temp_state_dict[name].copy_(param)
            
            temp_model.load_state_dict(temp_state_dict)
            
            # 3. Evaluate the temporary model on the client's test set.
            test_loss, test_acc = evaluate(temp_model, data['test_loader'], CRITERION, device)
            
            metric_line = (f"  > GLOBAL E-{global_epoch} | Test on C-{client_name} "
                           f"Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
            global_epoch_metrics.append(metric_line)
            save_log(metric_line)
            
            # Store the metrics for the final summary table
            if global_epoch not in global_test_results: global_test_results[global_epoch] = {}
            global_test_results[global_epoch][client_name] = {'loss': test_loss, 'acc': test_acc}
            
        # Log all per-epoch client training details
        GLOBAL_LOGS.extend(local_logs_per_epoch)
        GLOBAL_LOGS.extend(global_epoch_metrics)

    # --- [4] Final Model Save ---
    model_save_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    # The saved global model has the max_classes output size
    torch.save(global_model.state_dict(), model_save_path) 
    save_log(f"\n--- [4.0] Final Global Model Saved at: {model_save_path} ---")

    # --- [5] Final Evaluation and Reporting ---
    final_reports = []
    print(f"\n--- [5.0] Running final prediction on all test sets using global model ---")
    for client_name, data in client_data.items():
        # Final test model preparation
        final_temp_model = create_model(data['num_classes'])
        global_state_dict = global_model.state_dict()
        final_temp_state_dict = final_temp_model.state_dict()

        # Use the same filtering logic for the final report
        for name, param in global_state_dict.items():
             if 'classifier.3' not in name:
                if name in final_temp_state_dict and param.shape == final_temp_state_dict[name].shape:
                    final_temp_state_dict[name].copy_(param)
        
        final_temp_model.load_state_dict(final_temp_state_dict)

        report = final_predict_report(
            final_temp_model, 
            data['test_loader'], 
            data['class_names'], 
            client_name
        )
        final_reports.append(report)

    # --- [6] Write Results File ---
    print(f"\n--- [6.0] Writing all results to {results_path} ---")
    
    # Prepare the per-epoch tracking table
    tracking_table = (
        "\n======================================================\n"
        "PER GLOBAL EPOCH TEST ACCURACY TRACKING\n"
        "======================================================\n"
        f"Global Epoch (Total: {GLOBAL_EPOCHS}) | HAM Test Acc | ISIC Test Acc | PAD Test Acc\n"
        f"------------------------------------------------------\n"
    )
    for epoch in range(1, GLOBAL_EPOCHS + 1):
        ham_acc = global_test_results[epoch].get('ham', {}).get('acc', 'N/A')
        isic_acc = global_test_results[epoch].get('isic', {}).get('acc', 'N/A')
        pad_acc = global_test_results[epoch].get('pad', {}).get('acc', 'N/A')
        
        tracking_table += (
            f"E-{epoch:02d} | "
            f"{ham_acc:.2f}% | "
            f"{isic_acc:.2f}% | "
            f"{pad_acc:.2f}%\n"
        )
    tracking_table += "------------------------------------------------------\n"

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("SkinLesNet FEDERATED LEARNING (FedAvg) RESULTS\n")
        f.write(f"Start Time: {time.ctime()}\n")
        f.write(f"Global Epochs: {GLOBAL_EPOCHS}, Local Epochs: {LOCAL_EPOCHS}, LR: {LEARNING_RATE}\n")
        f.write("\n" + tracking_table + "\n")
        
        # Write final reports
        for report in final_reports:
            f.write(report + "\n")
        
        f.write("\n======================================================\n")
        f.write("DETAILED PER-EPOCH LOGS (Train & Test Metrics)\n")
        f.write("======================================================\n")
        for log in GLOBAL_LOGS:
            f.write(log + "\n")
            
        f.write("\nModel Path:\n")
        f.write(model_save_path + "\n")
        f.write(f"\nEnd Time: {time.ctime()}\n")

    save_log(f"--- [6.1] All data and model saved successfully. ---")

if __name__ == '__main__':
    run_fedavg()