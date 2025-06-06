import os
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Common transform: convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Base directory to save images
base_dir = './cifar10_images'

# Function to save dataset (train/test)
def save_cifar10_images(train: bool, output_subdir: str):
    # Download and load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    # Output path
    output_dir = os.path.join(base_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Save each image to a class folder
    for i, (img_tensor, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {output_subdir}"):
        class_name = dataset.classes[label]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        save_image(img_tensor, os.path.join(class_dir, f'{i}.png'))

# Save training images
save_cifar10_images(train=True, output_subdir='train')

# Save test images
save_cifar10_images(train=False, output_subdir='test')

print("✅ All CIFAR-10 train and test images saved successfully.")