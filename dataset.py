import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SalienceDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Assuming mask has same name

        # Handle case where mask might have different extension (e.g. .png vs .jpg)
        # For now assuming exact match or user ensures it. 
        # If file not found, try replacing extension or similar could be added.
        if not os.path.exists(mask_path):
             # Try replacing extension with .png if original is .jpg
             base, _ = os.path.splitext(img_name)
             mask_path = os.path.join(self.mask_dir, base + ".png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale for mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
