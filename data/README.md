# Data Folder – Structure & Preprocessing Guide

This folder contains the entire dataset structure used for training, testing, benchmarking, and evaluating the image sharpening pipeline. It includes both sharp and blurry patches, derived from the original high-resolution DIV2K dataset.

All patches are stored as `.jpg` images in 512×512 resolution.

---

## Dataset Source & Generation

* **Original Source**: [DIV2K\_train\_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* **Patch Size**: 512×512, non-overlapping
* **Blurry Generation**: Downscale to 128×128 → Upscale back to 512×512 using bicubic interpolation
* **Format**: `.jpg` images

### Patch Count Summary

| Folder                | Purpose                   | Approx. Count |
| --------------------- | ------------------------- | ------------- |
| `whole_dataset/`      | Raw DIV2K HR images       | 800           |
| `sharp/train/train/`  | Ground truth patches      | \~20,000      |
| `sharp/train/test/`   | Ground truth (test split) | \~5,000       |
| `sharp/benchmark/`    | Ground truth (benchmark)  | \~2,800       |
| `blurry/train/train/` | Blurry input patches      | \~20,000      |
| `blurry/train/test/`  | Blurry test inputs        | \~5,000       |
| `blurry/benchmark/`   | Blurry benchmark inputs   | \~2,800       |

---

## Folder Structure

```
data/
├── whole_dataset/             # Original HR images from DIV2K
├── blurry/
│   ├── train/
│   │   ├── train/             # 80% of 90% (train)
│   │   └── test/              # 20% of 90% (test)
│   └── benchmark/             # 10% benchmark
├── sharp/
│   ├── train/
│   │   ├── train/             # Ground truth for training
│   │   └── test/              # Ground truth for testing
│   └── benchmark/             # Ground truth for benchmark
```

> Each corresponding blurry and sharp image shares the same filename suffix (`0001_0_0.jpg`, etc.) for perfect alignment.

---

## How to Generate Dataset

Run the `dataset preprocessing` block from the main notebook:

* Loads original DIV2K `.png` files
* Splits: 90% train/test, 10% benchmark
* Applies non-overlapping 512×512 cropping
* Applies blur transformation (down+up) for blurry images
* Saves sharp and blurry images to structured directories in `.jpg` format

All data is saved permanently in your Google Drive inside:

```
/content/drive/MyDrive/ImageSharpening_KD/data/
```

---

## Quality Control

To verify correctness:

* File count is printed per directory
* All folders checked for emptiness
* Naming ensures patch alignment across sharp, blurry, and teacher images

> You can run the verification cell from Day 1 to check this:

```python
# Sample: check if each folder contains images
os.listdir('data/sharp/train/test/')
```

---

## Best Practices

* Use JPEG quality 90+ for storing patches to balance quality vs. size
* Avoid overlapping patches to prevent label leakage in training
* Keep raw `whole_dataset/` untouched for reproducibility
* Patch size of 512×512 is optimized for SwinIR and Mini-UNet

---

## Notes

* Dataset is **not included** in the GitHub repository due to size constraints
* Folder structure is maintained using `.gitkeep` or `README.md` placeholders for reproducibility
* You may recreate the dataset using the original DIV2K `.png` images if needed

---

## Maintainers

* **Dhruv** — Dataset preparation and quality checks
* **Pratham Patel** — Patch generation and structure planning

---

## License

All derived patches follow the license of the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Only for academic and research use.

---

