# Retinal Image Analysis Exploratory Project

This repository is a **small exploratory demo** for retinal image analysis.  
It includes two independent pipelines:

1. **Retinal image classification** (fundus disease category prediction)
2. **Retinal vessel segmentation** (vessel mask prediction + simple biomarker extraction)

The goal is to demonstrate end-to-end workflow understanding (data handling, model training, inference, visualization), **not** to claim benchmark-level performance.

---

## 1) Repository Structure

```text
Retinal_Image_Analysis/
├── classification/
│   ├── train_model.py                    # Train classification model (SqueezeNet transfer learning)
│   ├── predict.py                        # Single-image inference for classification
│   ├── sample_data.py                    # Randomly sample a small subset from original classification dataset
│   ├── utils.py                          # Data transforms and class name utilities
│   ├── retinal_classification_demo.ipynb # Notebook demo for classification / visualization
│   └── outputs/                          # Saved checkpoints + TensorBoard logs
│
├── segmentation/
│   ├── train_model.py                    # Train segmentation model (U-Net)
│   ├── evaluate.py                       # Test-time evaluation + visualization + biomarker extraction
│   ├── model.py                          # U-Net model builder
│   ├── data_loader.py                    # DRIVE dataset loader + preprocessing transforms
│   ├── utils.py                          # Dice/IoU metrics, biomarkers, plotting helpers
│   ├── retinal_seg_demo.ipynb            # Notebook demo for segmentation / visualization
│   └── outputs/                          # Saved checkpoints + evaluation figures
│
├── requirements.txt                      # Python dependencies
└── README.md                             # English README (this file)
```

---

## 2) Environment & Dependencies

### Recommended environment
- Python 3.9+
- PyTorch + torchvision
- segmentation_models_pytorch
- scikit-image
- matplotlib
- jupyter / notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` is a full environment export (including many notebook-related packages). You can also create a lightweight environment with only the core packages above.

---

## 3) Data Overview

### Classification data (FundusImage1000)
- **Source**: Kaggle dataset page  
  https://www.kaggle.com/datasets/linchundan/fundusimage1000
- **Scale**: 1000 fundus images, 39 disease/condition classes.
- **Collection context**: images from Joint Shantou International Eye Centre (JSIEC), Shantou, Guangdong, China; this subset comes from a larger 209,494-image pool used for deep-learning development.
- **Project usage**: `sample_data.py` creates a small demo subset in `classification/datasets/sampled_images` (default: 3 images per class).
- **Citation**:
  - "Automatic detection of 39 fundus diseases and conditions in retinal photographs using deep neural networks"  
    https://www.nature.com/articles/s41467-021-25138-w

### Segmentation data (DRIVE: Digital Retinal Images for Vessel Extraction)
- **Source**:
  - Kaggle mirror: https://www.kaggle.com/datasets/zionfuo/drive2004
  - Official site: https://drive.grand-challenge.org/
- **Scale**:
  - Total: 40 color fundus images.
  - Split: 20 training images + 20 test images.
  - Clinical composition: 33 without visible DR signs, 7 with mild early diabetic retinopathy.
- **Image characteristics**:
  - Camera/FOV: Canon CR5 non-mydriatic 3CCD camera, 45° field of view.
  - Resolution: 768×584, 8 bits per color channel.
  - Circular FOV (~540 px diameter) with provided FOV mask for each image.
- **Annotation/evaluation context**:
  - Training set includes manual vessel annotations (`1st_manual`) and mask files.
  - The project uses DRIVE folder structure: `training/images`, `training/1st_manual`, `training/mask`, `test/images`, `test/1st_manual`, `test/mask`.
  - In this repo, `train_model.py` further splits DRIVE training data into train/validation.
- **Why this dataset is used**: DRIVE is a standard benchmark for retinal vessel segmentation and morphology analysis (length, width, tortuosity, branching), relevant to ophthalmic and cardiovascular screening research.

> This project is demo-oriented and intentionally uses lightweight settings / limited epochs.

---

## 4) Models and Methods

### A. Classification pipeline
- **Backbone**: `torchvision.models.squeezenet1_1(pretrained=True)`
- **Strategy**:
  - Freeze pretrained feature extractor
  - Replace final classifier conv layer to match custom class count
  - Train only classifier head with cross-entropy loss
- **Input preprocessing**:
  - Train: random resized crop + horizontal flip + ImageNet normalization
  - Val/Test: resize + center crop + ImageNet normalization

### B. Segmentation pipeline
- **Backbone model**: U-Net from `segmentation_models_pytorch`
- **Encoder**: `resnet18` with ImageNet pretrained weights
- **Output**: 1-channel vessel mask with sigmoid activation
- **Loss**: Dice loss + BCE loss (combined)
- **Metrics**: Dice, IoU
- **Extra analysis**:
  - Vessel density within FOV
  - Skeleton length (using `skimage.morphology.skeletonize`)

---

## 5) Execution Flow (How the code runs)

### Classification workflow
1. (Optional) Build a tiny demo subset:
   ```bash
   cd classification
   python sample_data.py
   ```
2. Train model:
   ```bash
   python train_model.py
   ```
   - Creates timestamped folder in `classification/outputs/`
   - Saves `best_model.pth`, TensorBoard logs, and `class_names.json`
3. Predict one image:
   ```bash
   python predict.py --image path/to/image.jpg
   ```
   - If `--model` is omitted, script auto-selects latest checkpoint
   - Prints Top-K predicted classes
4. Notebook demo:
   - Open `classification/retinal_classification_demo.ipynb`

### Segmentation workflow
1. Train model:
   ```bash
   cd segmentation
   python train_model.py
   ```
   - Creates timestamped folder in `segmentation/outputs/`
   - Saves best checkpoint by validation Dice
2. Evaluate on test set:
   ```bash
   python evaluate.py
   ```
   - If no model path provided, auto-selects latest checkpoint
   - Computes Dice / IoU
   - Saves comparison plots under `test_results/`
   - Prints average vessel density and skeleton length
3. Notebook demo:
   - Open `segmentation/retinal_seg_demo.ipynb`
   - ![Segmentation Example](https://github.com/ChenYuze-JLG/Retinal_Image_Analysis/blob/main/segmentation/outputs/20260307-230417/test_results/comparison_01.png)
---

## 6) Highlights & Learning Outcomes

- Implemented both classification and segmentation retinal pipelines with PyTorch ecosystem.
- Practiced end-to-end engineering steps: preprocessing, training, evaluation, checkpointing, and TensorBoard logging.
- Added interpretable post-analysis in segmentation (vessel density + skeleton length).
- Prepared notebook demos for clear, visual presentation of classification and segmentation results.

---

## 7) Disclaimer

- This repository is an **exploratory demo**, not a clinical or benchmark system.
- Training settings are intentionally lightweight; metrics are not meant to represent state-of-the-art performance.
- Main value: showing practical understanding of retinal image analysis workflows.
