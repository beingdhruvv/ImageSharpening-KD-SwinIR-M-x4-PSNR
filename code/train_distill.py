import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from student_model import MiniUNet
from dataset_loader import DistillDataset

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

# === Dataset (Full) ===
dataset = DistillDataset(blurry_dir, sharp_dir, teacher_dir)  # ✅ Full dataset
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# === Model, Optimizer, Loss ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniUNet().to(device)

l1_loss = nn.L1Loss()
distill_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

alpha = 0.7  # GT supervision
beta = 0.3   # Teacher distillation

# === Training Loop (25 Epochs)
log_file = os.path.join(log_dir, "student_kd_training_log.txt")
with open(log_file, "w") as log:
    for epoch in range(75):  # ✅ Increased epochs
        total_loss = 0
        model.train()
        for blurry, sharp, teacher in loader:
            blurry = blurry.to(device)
            sharp = sharp.to(device)
            teacher = teacher.to(device)

            optimizer.zero_grad()
            output = model(blurry)

            loss_gt = l1_loss(output, sharp)
            loss_teacher = distill_loss(output, teacher)
            total = alpha * loss_gt + beta * loss_teacher

            total.backward()
            optimizer.step()
            total_loss += total.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/75 - Total Loss: {avg_loss:.4f}")
        log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}\n")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(model_dir, f'student_kd_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

# === Save Final Model
torch.save(model.state_dict(), os.path.join(model_dir, 'student_kd.pt'))
print("✅ Final model saved.")

# === Save Sample Output
model.eval()
with torch.no_grad():
    blurry, _, _ = next(iter(loader))
    pred = model(blurry.to(device)).clamp(0, 1)
    save_image(pred, os.path.join(output_dir, 'student_kd_sample.jpg'))
    print("🖼️ Saved sample student output.")
