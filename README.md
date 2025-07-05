# Image Sharpening using Knowledge Distillation (SwinIR Teacher)

This repository contains a complete end-to-end pipeline for **image sharpening (deblurring)** using **knowledge distillation**. A powerful SwinIR-M model acts as the **teacher**, while a lightweight **Mini-UNet** is trained as the **student** to mimic its behavior. The student model is capable of real-time performance while retaining strong SSIM/PSNR quality.

---

## Project Summary

| Component      | Details                                                              |
| -------------- | -------------------------------------------------------------------- |
| Teacher     | SwinIR-M PSNR-optimized model (`x4`, trained on BSRGAN degradations) |
| Student     | Mini-UNet (2–3 stages, skip connections, Conv+BN+ReLU blocks)        |
| Patch Size | 512×512                                                              |
| Goal        | SSIM ≥ 0.90 on benchmark/test via distillation training              |
| Platform    | Google Colab + Google Drive (Free GPU Tier)                          |

---

## Folder Structure (with placeholders)

This project is structured to keep the dataset excluded, but its structure preserved with `.gitkeep` or `README.md` placeholders:

```
ImageSharpening_KD/
├── code/                        # All training, testing, model scripts
│   ├── student_model.py         # Mini-UNet architecture
│   ├── dataset_loader.py        # Dataset class with blurry, sharp, teacher triplets
│   ├── train_student.py         # L1-only training
│   ├── train_distill.py         # KD training (L1 + teacher output loss)
│   ├── train_distill_vgg.py     # Optional: adds perceptual (VGG) loss
│
├── models/                      # Teacher + student weights
│   ├── swinir_teacher/          # SwinIR pretrained model (README/placeholder only)
│   │   └── README.md            # Path/URL reference to model (not included)
│   └── student_kd.pt            # Final trained student model (if <100MB)
│
│
├── logs/                        # Training logs
│   └── student_kd_training_log.txt
│
├── data/                        # [Excluded from repo] Full structure maintained
│   ├── whole_dataset/.gitkeep   # Original DIV2K images (source only)
│   ├── blurry/train/train/.gitkeep
│   ├── blurry/train/test/.gitkeep
│   ├── blurry/benchmark/.gitkeep
│   ├── sharp/train/train/.gitkeep
│   ├── sharp/train/test/.gitkeep
│   └── sharp/benchmark/.gitkeep
│
├── ISKD - SwinIR-M_x4_PSNR.ipynb  # Final working Colab notebook (end-to-end)
└── README.md                     # This documentation
```

---

## Workflow Overview

### Dataset Preparation

* Source: `DIV2K_train_HR/` → saved in `/data/whole_dataset/`
* 512×512 patches created with non-overlapping stride
* Blurry patches created by downscaling + upscaling
* Splits:

  * **90% Training (→ 80% train, 20% test)**
  * **10% Benchmark**
* Total patches: `~55,916`

### Teacher Inference (SwinIR)

* Model: `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth`
* Inference over all blurry patches
* Output stored in:

  * `/outputs/teacher_output/train/train/`
  * `/outputs/teacher_output/train/test/`
  * `/outputs/teacher_output/benchmark/`

### Student Training

* Architecture: Mini-UNet (2–3 downsampling blocks + skip connections)
* Loss Functions:

  * `loss_total = α * L1(output, GT_sharp) + β * MSE(output, Teacher_pred)`
  * Optional: add `+ γ * VGG(output, GT_sharp)`
* Training runs for 25–75 epochs depending on image count

### Evaluation

* SSIM + PSNR computed on all sets
* External test support (3–5 real-world blurry/sharp images)

---

## Results

| Dataset Split  | SSIM   | PSNR    |
| -------------- | ------ | ------- |
| Train          | 0.72   | 24.6 dB |
| Test           | 0.78   | 28.5 dB |
| Benchmark      | 0.75   | 26.5 dB |
| External (avg) | \~0.71 | -       |

> Note: SSIM calculated on full RGB uint8 images using scikit-image

---

## External Image Testing

* Test real-world image pairs via `test_external_images.py`
* Folder structure:

```
outputs/external_test/
├── input_blurry/
├── ground_truth/
├── predicted/
└── ssim_results.txt
```

* Auto-resumes and skips already tested images
* Displays visual comparison and prints SSIM score per image

---

## Requirements

This project is designed to run entirely in **Google Colab**. If running locally, install:

```bash
pip install torch torchvision timm einops yacs scikit-image opencv-python tqdm matplotlib
```

Optional:

* VGG-based perceptual loss (torchvision >= 0.15 recommended)

---

## How to Use

### Run in Colab

> Open the notebook:

```bash
ISKD - SwinIR-M_x4_PSNR.ipynb
```

Steps included:

1. Mount Drive
2. Preprocess & split dataset
3. Run SwinIR inference (teacher)
4. Train Mini-UNet (student)
5. Evaluate and visualize results

### Manual Script Execution

> Located in `/code/`

* `train_student.py`: L1 only
* `train_distill.py`: L1 + distillation loss
* `train_distill_vgg.py`: Adds VGG perceptual loss
* `test_external_images.py`: Evaluate 3–5 blurry-sharp pairs

---

## Model Weights

| Model               | Location                  | Size (approx.) |
| ------------------- | ------------------------- | -------------- |
| SwinIR-M (PSNR)     | `/models/swinir_teacher/` | \~47 MB        |
| Student (Mini-UNet) | `/models/student_kd.pt`   | \~1–5 MB       |

---

## Notes

* Dataset is excluded from this repo (too large) → use placeholders
* To recreate patches, refer to the preprocessing script in the notebook
* External test pairs should be visually and resolution matched

---

## Author & Credits

**Project Lead**: [Dhruv Suthar](https://github.com/beingdhruvv)
Project Co-developer: [Pratham Patel](https://github.com/prathampatel10)
**Inspired by**:

* [SwinIR](https://github.com/JingyunLiang/SwinIR)
* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

---

## License

MIT License. This project is open-source and free for academic, educational, and research use.

---

> For questions, raise an issue or submit a PR. Contributions welcome!
