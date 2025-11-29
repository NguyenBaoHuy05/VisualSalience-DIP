import os
import torch
from torchvision import transforms
from PIL import Image
from model import UNet
import matplotlib.pyplot as plt

def test_single_image(image_path, model_path, output_path='result.png', device='cuda'):
    # Load model
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    # Save/Show result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title("Predicted Saliency")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    # Example usage
    # You can change these paths
    img_path = 'images/test_image.jpg' # Replace with actual image path
    ckpt_path = 'model.pth'
    
    if os.path.exists(img_path) and os.path.exists(ckpt_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_single_image(img_path, ckpt_path, device=device)
    else:
        print(f"Please ensure {img_path} and {ckpt_path} exist.")
