import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from student_model import MiniUNet
from dataset_loader import ImagePairDataset

# Paths
project_root = '/content/drive/MyDrive/ImageSharpening_KD'
blurry_dir = os.path.join(project_root, 'data/blurry/train/train')
sharp_dir = os.path.join(project_root, 'data/sharp/train/train')
output_dir = os.path.join(project_root, 'outputs')
model_dir = os.path.join(project_root, 'models')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Dataset
dataset = ImagePairDataset(blurry_dir, sharp_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# Training
model.train()
for epoch in range(50):
    epoch_loss = 0
    for blurry, sharp in loader:
        blurry, sharp = blurry.to(device), sharp.to(device)
        optimizer.zero_grad()
        output = model(blurry)
        loss = criterion(output, sharp)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/5 - Avg L1 Loss: {epoch_loss / len(loader):.4f}")

# Save model
torch.save(model.state_dict(), os.path.join(model_dir, 'student_l1.pt'))

# Save output
model.eval()
with torch.no_grad():
    blurry, _ = next(iter(loader))
    blurry = blurry.to(device)
    pred = model(blurry).clamp(0, 1)
    save_image(pred, os.path.join(output_dir, 'student_sample.jpg'))
