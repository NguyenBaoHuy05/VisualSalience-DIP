import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=5, device='cuda', lr=1e-4, checkpoint_path='model.pth'):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(1)
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved to {checkpoint_path}')
