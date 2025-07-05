from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class ImagePairDataset(Dataset):
    def __init__(self, blurry_dir, sharp_dir, max_images=None):
        self.filenames = sorted([f for f in os.listdir(blurry_dir) if f.endswith('.jpg')])
        if max_images:
            self.filenames = self.filenames[:max_images]
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        blurry = Image.open(os.path.join(self.blurry_dir, name)).convert("RGB")
        sharp = Image.open(os.path.join(self.sharp_dir, name)).convert("RGB")
        return self.transform(blurry), self.transform(sharp)

class DistillDataset(Dataset):
    def __init__(self, blurry_dir, sharp_dir, teacher_dir, max_images=None):
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.teacher_dir = teacher_dir
        self.transform = transforms.ToTensor()
        self.resize_teacher = transforms.Resize((256, 256))  # 👈 Added

        blurry_files = set(os.listdir(blurry_dir))
        sharp_files = set(os.listdir(sharp_dir))
        teacher_files = set(os.listdir(teacher_dir))

        self.filenames = sorted(list(blurry_files & sharp_files & teacher_files))
        if max_images:
            self.filenames = self.filenames[:max_images]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        blurry = Image.open(os.path.join(self.blurry_dir, name)).convert("RGB")
        sharp = Image.open(os.path.join(self.sharp_dir, name)).convert("RGB")
        teacher = Image.open(os.path.join(self.teacher_dir, name)).convert("RGB")

        # 🧠 Resize teacher to 256x256 before converting to tensor
        teacher = self.resize_teacher(teacher)

        return self.transform(blurry), self.transform(sharp), self.transform(teacher)



