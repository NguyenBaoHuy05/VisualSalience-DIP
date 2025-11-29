
import torch
import random
from model import UNet

def test_sizes():
    model = UNet(n_channels=3, n_classes=1)
    model.eval()
    
    sizes = [
        (1, 3, 572, 572), # Standard UNet paper size
        (1, 3, 256, 256), # Power of 2
        (1, 3, 255, 255), # Odd size
        (1, 3, 100, 100), # Non-power of 2
        (1, 3, 32, 32),   # Small
        (1, 3, 16, 16),   # Minimum likely
        (1, 3, 15, 15),   # Smaller than factor 16?
        (1, 3, 300, 200), # Rectangular
        (1, 3, 200, 300), # Portrait Rectangular
        (1, 3, 131, 131), # Prime Square
        (1, 3, 131, 200), # Prime Width
        (1, 3, 101, 53),  # Prime Rectangular
        (1, 3, 53, 101),  # Prime Rectangular Portrait
        (1, 3, 1000, 50), # Extreme Aspect Ratio
        (1, 3, 50, 1000), # Extreme Aspect Ratio Portrait
    ]
    
    # Add some random rectangular sizes
    random.seed(42)
    for _ in range(5):
        h = random.randint(16, 500)
        w = random.randint(16, 500)
        sizes.append((1, 3, h, w))

    print(f"Testing {len(sizes)} cases...")
    failed = 0
    for size in sizes:
        try:
            x = torch.randn(*size)
            y = model(x)
            # Check batch, height, width match. Channels might differ.
            if y.shape[0] != size[0] or y.shape[2] != size[2] or y.shape[3] != size[3]:
                print(f"Input: {size} - FAILED: Output shape mismatch {y.shape}")
                failed += 1
            else:
                print(f"Input: {size} - OK")
        except Exception as e:
            print(f"Input: {size} - FAILED: {e}")
            failed += 1
            
    if failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failed} tests failed.")
        exit(1)

if __name__ == "__main__":
    test_sizes()
