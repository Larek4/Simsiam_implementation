# SimSiam for Earth Observation: Self-Supervised Learning for Land Cover Classification

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Project Structure](#2-project-structure)
3.  [Setup & Installation](#3-setup--installation)
4.  [Dataset Preparation](#4-dataset-preparation)
5.  [Usage & Reproduction](#5-usage--reproduction)
6.  [Key Results](#6-key-results)
7.  [Acknowledgements](#7-acknowledgements)

---

## 1. Introduction

This project implements a self-supervised learning (SSL) pipeline using the **SimSiam** method to pre-train a robust feature extractor for Earth Observation (EO) imagery. The goal is to learn general-purpose visual representations from unlabeled satellite data, significantly reducing the reliance on extensive labeled datasets. The pre-trained model is then adapted via **linear evaluation** and **full fine-tuning** for **land cover classification** on the EuroSAT RGB dataset.

## 2. Project Structure

```
.
├── barlow_data/               # Converted RGB PNGs for SimSiam pre-training input.
├── data/                      # EuroSAT RGB dataset (class folders, CSVs, label_map.json) for fine-tuning/evaluation.
│   └── EuroSAT/
├── simsiam_data.py            # PyTorch Dataset and data augmentation pipelines for SimSiam.
├── simsiam_model.py           # SimSiam model architecture (ResNet-50 backbone, Projection, and Prediction heads).
├── train_simsiam.py           # Main script for SimSiam self-supervised pre-training.
├── Linear_evaluation.py       # Script for linear classification evaluation on EuroSAT (backbone frozen).
├── finetune_model.py          # Script for full fine-tuning on EuroSAT (backbone unfrozen).
├── classify_image.py          # Utility script to classify a single image.
├── plot_results.py            # (To be created) Script to generate performance graphics.
├── .gitignore                 # Specifies intentionally untracked files (e.g., large datasets, environments).
└── README.md                  # This README file.
```

## 3. Setup & Installation

This project was developed on a system with **NVIDIA Quadro RTX 4000 (8 GB VRAM)** and **20 CPU Cores**.


1.  **Create Conda Environment:**
    ```bash
    conda create -n ssl4eo_env python=3.10
    conda activate ssl4eo_env
    ```

3.  **Install Libraries:**
    ```bash
    pip install torch torchvision numpy pillow pandas scikit-learn tqdm
    pip install rasterio # Required for specific data handling (if using zarr_to_images.py)
    ```

## 4. Dataset Preparation

This project uses two main datasets:

* **Pre-training Data (SSL4EO-S12):**
    * Derived from large `.zarr.zip` archives (not included in this repository due to size).
    * **Requires conversion:** Multi-band `.zarr` files must be converted to 3-channel RGB PNG images (e.g., using Sentinel-2's B4, B3, B2 bands).
    * **Placement:** Place the converted PNGs into the `barlow_data/train` directory.
    * *Note: A custom conversion script (`zarr_to_images.py`) used during development is not provided here.*

* **Fine-tuning Data (EuroSAT RGB):**
    * Download the EuroSAT RGB dataset (e.g., from Kaggle).
    * It should contain 10 class subdirectories, `train.csv`, `validation.csv`, `test.csv`, and `label_map.json`.
    * **Placement:** Place the extracted `EuroSAT` folder inside your project's `data/` directory (e.g., `data/EuroSAT`).

## 5. Usage & Reproduction

Ensure your `ssl4eo_env` is activated and datasets are prepared as described above. Navigate to the root directory of this repository.

1.  **Pre-train SimSiam Model:**
    ```bash
    python train_simsiam.py
    ```
    * (Approx. 5.5 minutes per epoch for 200 epochs on specified hardware. Checkpoints and `pretrain_metrics.pth` will be saved.)

2.  **Perform Linear Evaluation:**
    ```bash
    python Linear_evaluation.py
    ```
    * (Evaluates pre-trained backbone's features; `linear_eval_eurosat_best_model.pth` and `linear_eval_metrics.pth` saved.)

3.  **Perform Full Fine-tuning:**
    ```bash
    python finetune_model.py
    ```
    * (Fine-tunes the entire model for downstream task; `finetuned_eurosat_best_model.pth` and `finetune_metrics.pth` saved.)

4.  **Classify a Single Image:**
    * Update `IMAGE_TO_CLASSIFY_PATH` in `classify_image.py` first.
    ```bash
    python classify_image.py
    ```

5.  **Generate Performance Graphics:**
    * (Assumes `plot_results.py` is created to load `*.pth` metrics and visualize results.)
    ```bash
    python plot_results.py
    ```

## 6. Key Results

* **Linear Evaluation:** Achieved **95.81% Test Accuracy** on EuroSAT, demonstrating high quality of pre-trained features.
* **Full Fine-tuning:** Achieved a remarkable **98.63% Test Accuracy** on EuroSAT, showcasing the benefit of adapting the pre-trained backbone.

## 7. Acknowledgements

* **SimSiam:** Chen, T., et al. (2021). *SimSiam: Exploring Simple Siamese Representation Learning.* ICCV.
* **EuroSAT Dataset:** Helber, P., et al. (2019). *EuroSAT: A Novel Dataset for Deep Learning in Remote Sensing.* IEEE TGRS.
* **SSL4EO-S12 Dataset:** (Cite the specific SSL4EO-S12 paper/source if applicable)
```
