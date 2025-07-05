# Code Folder – Scripts for Training & Evaluation

This folder contains all the core scripts used to train and evaluate the student Mini-UNet model for image sharpening via knowledge distillation from the SwinIR teacher model.

---

## Script Descriptions

| File                     | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| `student_model.py`       | Defines a compact Mini-UNet architecture with 2–3 encoder-decoder stages.  |
| `dataset_loader.py`      | Loads blurry, sharp, and teacher output triplets for training and testing. |
| `train_student.py`       | Trains the student model using **only L1 loss** (supervised learning).     |
| `train_distill.py`       | Trains using both **L1 loss + MSE distillation loss** from the teacher.    |
| `train_distill_vgg.py`   | Trains using **L1 + MSE + perceptual loss** using a VGG feature extractor. |
| `evaluate_student_kd.py` | Evaluates student model performance using **SSIM and PSNR** metrics.       |

---

## How to Use the Scripts

> ⚠Before running any script, ensure dataset paths and output directories are correctly configured inside the files.

### Train the Student with L1 Loss Only

```bash
python train_student.py
```

### Train with L1 + Distillation Loss

```bash
python train_distill.py
```

### Train with L1 + Distillation + Perceptual Loss (VGG)

```bash
python train_distill_vgg.py
```

### Evaluate Student Model (SSIM/PSNR)

```bash
python evaluate_student_kd.py
```

---

## Tips

* All training is performed on 512×512 patches.
* Scripts are optimized for **Google Colab with GPU**.
* Logs and sample outputs are saved automatically to `/logs/` and `/outputs/`.
* Perceptual loss (VGG) requires `torchvision >= 0.15`.

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute it for academic and research purposes.

---
