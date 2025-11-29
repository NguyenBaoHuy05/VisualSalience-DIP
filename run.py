import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import SalienceDataset
from model import UNet
from train import train_model

def main():
    # Configuration
    image_dir = 'images'
    mask_dir = 'ground_truth_mask'
    batch_size = 4 # Small batch size for safety
    epochs = 5
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Dataset
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Error: Directories {image_dir} and/or {mask_dir} not found.")
        # Create dummy directories for demonstration if they don't exist
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        print("Created empty directories. Please populate them with images and masks.")
        return

    dataset = SalienceDataset(image_dir, mask_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No images found in dataset. Please add images to 'images/' and masks to 'ground_truth_mask/'.")
        return

    # Split
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = UNet(n_channels=3, n_classes=1)

    # Train
    train_model(model, train_loader, val_loader, epochs=epochs, device=device, lr=lr)

if __name__ == '__main__':
    main()
