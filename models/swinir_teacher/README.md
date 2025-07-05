# SwinIR Teacher Folder

This folder contains the pretrained **SwinIR-M PSNR-optimized teacher model** used for generating target outputs during knowledge distillation.

The model is used in inference mode only and **never trained or fine-tuned**. It provides high-quality sharpened outputs that guide the student Mini-UNet during training.

---

## Model File

```
swinir_teacher/
├── 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth
```

* Type: PSNR-optimized model for real-world SR x4 tasks
* Format: `.pth` PyTorch checkpoint
* Source: [SwinIR Official GitHub](https://github.com/JingyunLiang/SwinIR)
* File Size: \~47MB

> This file is too large for GitHub upload and should be downloaded manually or hosted externally.

---

## 🔧 How to Use

Use the SwinIR model during teacher inference stage:

```python
from basicsr.archs.swinir_arch import SwinIR
model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1., depths=[6,6,6,6], embed_dim=180, num_heads=[6,6,6,6], mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
model.load_state_dict(torch.load('models/swinir_teacher/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth'))
model.eval()
```

Used in:

* SwinIR inference over blurry patches
* `outputs/teacher_output/`: Stores teacher predictions

---

## Notes

* Model was trained using the **BSRGAN + RealSR** degradation pipeline
* Performs real-world super-resolution, but here used for **deblurring/sharpening** effect
* Ensure correct architecture instantiation before loading weights

---

## License

This model is provided under the original [SwinIR license](https://github.com/JingyunLiang/SwinIR/blob/main/LICENSE). Refer to their repo for citation and reuse terms.

---
