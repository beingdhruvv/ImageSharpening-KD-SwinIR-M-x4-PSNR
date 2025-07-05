# Pretrained & Trained Weights

This folder contains all model weights used in this project, including:

* The pretrained **SwinIR-M teacher model** (used for inference)
* The final trained **Mini-UNet student model**
* Optionally: intermediate checkpoints saved across epochs

All files are in `.pth` PyTorch format and should be loaded using `torch.load()`.

---

## Folder Structure

```
models/
├── swinir_teacher/
│   ├── 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth   # Teacher model
│   └── README.md (or .gitkeep placeholder)
├── student_kd.pt          # Final trained student (KD)
├── student_l1.pt          # Student trained with L1 only (optional)
├── student_kd_epoch25.pt  # Checkpoint at epoch 25 (optional)
└── ...                    # Any additional checkpoints
```

> SwinIR model is used only for inference. It is not fine-tuned or trained further.

---

## Model Types

| File                    | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| `swinir_teacher/*.pth`  | Used for generating teacher outputs      |
| `student_kd.pt`         | Final trained Mini-UNet (KD trained)     |
| `student_l1.pt`         | Student trained with only L1 loss        |
| `student_kd_epochXX.pt` | Optional checkpoint saved every N epochs |

---

## Usage Example

To load a model checkpoint in your script:

```python
import torch
from student_model import MiniUNet

model = MiniUNet()
model.load_state_dict(torch.load('models/student_kd.pt'))
model.eval()
```

> Ensure model architecture matches saved checkpoint structure.

---

## Where Files Are Saved

All model checkpoints are saved automatically by training scripts to:

```
/content/drive/MyDrive/ImageSharpening_KD/models/
```

You may manually copy pretrained models here or download from official sources.

---

## GitHub & File Size

* Student models (`.pt`) under 100MB are included in the repo
* Teacher model is **not uploaded** due to size; placeholder or download link is added in `README.md`
* Intermediate checkpoints may be excluded unless essential


---

## License

All student models are released under the project MIT License. SwinIR teacher model is referenced from [SwinIR GitHub](https://github.com/JingyunLiang/SwinIR) and follows its original license.

---

