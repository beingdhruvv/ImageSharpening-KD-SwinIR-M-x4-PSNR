import os
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from student_model import MiniUNet
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

# === Paths ===
project_root = '/content/drive/MyDrive/ImageSharpening_KD'
blurry_dir = os.path.join(project_root, 'data/blurry/benchmark')
sharp_dir = os.path.join(project_root, 'data/sharp/benchmark')
output_dir = os.path.join(project_root, 'outputs/student_kd_output')
model_path = os.path.join(project_root, 'models/student_kd_vgg.pt')
log_path = os.path.join(project_root, 'logs')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

to_tensor = ToTensor()
to_pil = ToPILImage()

# === Inference & Evaluation ===
ssim_total = 0
psnr_total = 0
count = 0

filenames = sorted([f for f in os.listdir(blurry_dir) if f.endswith('.jpg')])

for fname in tqdm(filenames, desc="Evaluating"):
    blurry_img = Image.open(os.path.join(blurry_dir, fname)).convert("RGB")
    sharp_img = Image.open(os.path.join(sharp_dir, fname)).convert("RGB")

    input_tensor = to_tensor(blurry_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze().cpu().clamp(0, 1)

    output_img = to_pil(output_tensor)
    output_img.save(os.path.join(output_dir, fname), quality=95)

    # Resize sharp image to match (in case needed)
    sharp_img = sharp_img.resize(output_img.size, Image.BICUBIC)

    out_np = np.array(output_img)
    sharp_np = np.array(sharp_img)

    ssim = compare_ssim(out_np, sharp_np, channel_axis=-1)
    psnr = compare_psnr(sharp_np, out_np)

    ssim_total += ssim
    psnr_total += psnr
    count += 1

# === Results ===
avg_ssim = ssim_total / count
avg_psnr = psnr_total / count

print(f"\n📊 Student Evaluation on Benchmark:")
print(f"✅ SSIM: {avg_ssim:.4f}")
print(f"✅ PSNR: {avg_psnr:.2f} dB")

# Save log
with open(os.path.join(log_path, 'student_kd_scores.txt'), 'w') as f:
    f.write(f"SSIM: {avg_ssim:.4f}\nPSNR: {avg_psnr:.2f} dB\n")
