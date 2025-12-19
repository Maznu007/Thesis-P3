import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Step 1: Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resize all images to the same size
    transforms.ToTensor()
])

# Step 2: Load dataset from your PC
data_dir = r"D:\dataset\pad\organized_pad\test"  # <-- your dataset path
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Get class names
class_names = dataset.classes
print("Classes found:", class_names)

# Step 3: Pick one sample per class
samples = {}
for img, label in dataset:
    if label not in samples:
        samples[label] = img
    if len(samples) == len(class_names):
        break

# Step 4: Plot samples
n_classes = len(class_names)
fig, axes = plt.subplots(1, n_classes, figsize=(3*n_classes, 3))

# Handle single-class edge case
if n_classes == 1:
    axes = [axes]

for idx, (label, img) in enumerate(samples.items()):
    axes[idx].imshow(np.transpose(img.numpy(), (1, 2, 0)))  # CHW -> HWC
    axes[idx].set_title(class_names[label], fontsize=10, pad=8)
    axes[idx].axis("off")

# Add a main title
fig.suptitle("PAD-UFES-20 Sample Class Example", fontsize=16, weight="bold")

# Adjust layout so title is visible
plt.tight_layout(rect=[0, 0, 1, 0.93])  # leaves space for suptitle
plt.show()

# Step 5: Save figure (for research paper)
fig.savefig("class_samples.png", dpi=300, bbox_inches="tight")
print("✅ Saved figure as class_samples.png")
