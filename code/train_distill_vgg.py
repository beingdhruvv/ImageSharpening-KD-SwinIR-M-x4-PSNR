import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
from student_model import MiniUNet
from dataset_loader import DistillDataset
import torchvision.transforms as transforms

# === Paths ===
project_root = '/content/drive/MyDrive/ImageSharpening_KD'
blurry_dir = os.path.join(project_root, 'data/blurry/train/train')
sharp_dir = os.path.join(project_root, 'data/sharp/train/train')
teacher_dir = os.path.join(project_root, 'outputs/teacher_output/train')
output_dir = os.path.join(project_root, 'outputs')
model_dir = os.path.join(project_root, 'models')
log_dir = os.path.join(project_root, 'logs')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# === Dataset ===
dataset = DistillDataset(blurry_dir, sharp_dir, teacher_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# === Load Pretrained VGG16 (perceptual loss)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]  # Up to conv3_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        return nn.functional.l1_loss(self.vgg(x), self.vgg(y))

# === Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniUNet().to(device)
l1_loss = nn.L1Loss()
distill_loss = nn.MSELoss()
vgg_loss = VGGPerceptualLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Loss weights
alpha = 0.6  # L1
beta = 0.2   # Distillation
gamma = 0.2  # Perceptual

# === Training Loop
log_file = os.path.join(log_dir, "student_kd_vgg_training_log.txt")
with open(log_file, "w") as log:
    for epoch in range(75):
        total_loss = 0
        model.train()

        for blurry, sharp, teacher in loader:
            blurry = blurry.to(device)
            sharp = sharp.to(device)
            teacher = teacher.to(device)

            optimizer.zero_grad()
            output = model(blurry)

            loss_l1 = l1_loss(output, sharp)
            loss_kd = distill_loss(output, teacher)
            loss_vgg = vgg_loss(output, sharp)

            total = alpha * loss_l1 + beta * loss_kd + gamma * loss_vgg

            total.backward()
            optimizer.step()
            total_loss += total.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/75 - Loss: {avg_loss:.4f}")
        log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}\n")

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(model_dir, f'student_kd_vgg_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

# === Save final model
torch.save(model.state_dict(), os.path.join(model_dir, 'student_kd_vgg.pt'))
print("✅ Final VGG+KD student model saved.")

# === Save a sample output
model.eval()
with torch.no_grad():
    blurry, _, _ = next(iter(loader))
    blurry = blurry.to(device)
    pred = model(blurry).clamp(0, 1)
    save_image(pred, os.path.join(output_dir, 'student_kd_vgg_sample.jpg'))
    print("🖼️ Saved sample student output (VGG).")
