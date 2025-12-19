import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import regnet_y_32gf
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for unique

def main():
    print("🚀 Starting RegNetY training on HAM dataset and testing on ISIC P3 dataset...")

    # --- Path Configuration ---
    print("📁 Setting paths...")
    
    # Training Data Path (original HAM dataset)
    train_base_dir = r"D:\dataset\ham\organized_ham"
    train_path = os.path.join(train_base_dir, "train")
    
    # Testing Data Path (new ISIC dataset)
    isic_base_dir = r"D:\dataset\isic\Skin cancer ISIC The International Skin Imaging Collaboration"
    test_path = os.path.join(isic_base_dir, "Test")
    
    # Results Save Path (new folder)
    results_dir = os.path.join(isic_base_dir, "isic_p3_result", "trainonhamtestonisic_p3")
    os.makedirs(results_dir, exist_ok=True) # Create the new directory if it doesn't exist

    # New file names
    model_save_path = os.path.join(results_dir, "regnety320_ham_to_isic_model.pth")
    results_txt_path = os.path.join(results_dir, "results_ham_to_isic.txt")
    conf_matrix_path = os.path.join(results_dir, "confusion_matrix_ham_to_isic.png")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # --- Data Loading and Transforms ---
    print("🌀 Preparing image transformations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load datasets
    print("📂 Loading training (HAM) and testing (ISIC) datasets...")
    # NOTE: train_dataset.classes will define the number of output classes and the labels for the confusion matrix.
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    print(f"✅ Loaded {len(train_dataset)} train and {len(test_dataset)} test images.")
    
    # Check for class consistency (Important when testing on a different dataset!)
    print(f"Training classes: {train_dataset.classes}") # Example: 7 classes
    print(f"Testing classes: {test_dataset.classes}") # Example: 9 classes
    
    # DataLoaders
    print("📦 Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- Model Setup ---
    print("🧠 Loading RegNetY-32GF model...")
    model = regnet_y_32gf(weights='DEFAULT')
    # The final layer is set to the number of classes in the *training* dataset (HAM).
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    # Loss and Optimizer
    print("⚙️  Setting loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Track metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    # --- Training Loop ---
    print("🏋️  Starting training loop...")
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0
        correct, total = 0, 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy for training
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 5 == 0:
                print(f"   Batch {i+1}: Loss = {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        # Validation/Test accuracy after each epoch
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_acc = correct_test / total_test
        history["test_acc"].append(test_acc)

        print(f"✅ Epoch {epoch+1} completed | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # --- Final Evaluation ---
    print("\n🔍 Final evaluation on ISIC test data...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # IMPORTANT: The model's output classes correspond to the *training* labels.
            # The confusion matrix will use the test labels and the train classes for interpretation.
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Convert to numpy arrays for sklearn
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # --- Metrics Calculation and Saving ---
    print("📊 Calculating metrics...")
    acc = accuracy_score(all_labels, all_preds)
    
    # --- FIX START ---
    # Determine all unique true labels present in the test set
    unique_test_labels = np.unique(all_labels)
    
    # Use the test dataset's classes as target_names, as the true labels (all_labels) are from the test set
    # The classification_report needs one name for every unique true label index.
    # The model predicts up to len(train_dataset.classes) - 1, but the true labels are from test_dataset.
    # We use test_dataset.classes for target_names and explicitly set the 'labels' parameter
    # to all unique true labels in the test set for a valid report.
    report_target_names = test_dataset.classes
    # The labels parameter specifies the set of labels to include in the report.
    report = classification_report(
        all_labels, 
        all_preds, 
        labels=unique_test_labels, # Only include true labels that were actually present
        target_names=report_target_names, 
        zero_division=0
    )
    
    # Note for confusion_matrix: It correctly handles labels not in test_dataset.classes if you pass 'labels'
    # For CM, we'll use all unique labels present in the combined set of true/predicted labels for rows/cols
    # and use the corresponding class names for display.
    # The number of rows will be unique true labels, the number of columns will be unique predicted labels.
    
    # The original CM call should be fine, but let's be explicit about the labels for robustness
    # The CM will have dimensions based on the unique true labels (rows) and unique predicted labels (cols).
    # Since the model only outputs len(train_dataset.classes), the columns will be 7. The rows can be up to 9.
    
    cm = confusion_matrix(
        all_labels, 
        all_preds, 
        labels=np.arange(len(test_dataset.classes)) # Use all possible ISIC labels 0 to 8
    ) 
    
    # --- FIX END ---
    
    # Save model
    print("💾 Saving model...")
    torch.save(model.state_dict(), model_save_path)

    # Save confusion matrix
    print("🖼️  Saving confusion matrix...")
    plt.figure(figsize=(12, 10))
    # Use the class names from the *testing* dataset for the y-axis (True Class)
    # The x-axis (Predicted Class) should technically only show the trained classes (HAM)
    # but using test_dataset.classes for both is generally clearer for a full report on the test set.
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=test_dataset.classes, # Show all possible ISIC classes on X-axis (Predictions)
        yticklabels=test_dataset.classes  # Show all possible ISIC classes on Y-axis (True Labels)
    )
    plt.title("Confusion Matrix (Trained on HAM, Tested on ISIC)")
    plt.xlabel(f"Predicted Class (Model trained on HAM: {train_dataset.classes})")
    plt.ylabel(f"True Class (from ISIC: {test_dataset.classes})")
    plt.tight_layout()
    plt.savefig(conf_matrix_path)
    plt.close()

    # Save results
    print("📄 Writing results to file...")
    with open(results_txt_path, "w") as f:
        f.write("--- RegNetY-32GF Transfer Learning Results ---\n")
        f.write(f"Training Dataset (HAM Classes: {len(train_dataset.classes)}):\n{train_dataset.classes}\n")
        f.write(f"Testing Dataset (ISIC Classes: {len(test_dataset.classes)}):\n{test_dataset.classes}\n\n")
        f.write(f"Final Test Accuracy: {acc:.4f}\n\n")
        f.write(f"Classification Report (Target Names are ISIC Classes - only true labels present: {unique_test_labels}):\n")
        f.write(report + "\n\n")

        f.write("Per-epoch results:\n")
        for i in range(len(history["train_loss"])):
            f.write(
                f"Epoch {i+1}: "
                f"Train Loss = {history['train_loss'][i]:.4f}, "
                f"Train Acc = {history['train_acc'][i]:.4f}, "
                f"Test Acc = {history['test_acc'][i]:.4f}\n"
            )

    print(f"\n✅ Training and evaluation complete. All results saved in: {results_dir}")

if __name__ == "__main__":
    main()


    #done