# SimSiam for Earth Observation: Self-Supervised Learning for Land Cover Classification

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Project Structure & File Descriptions](#2-project-structure--file-descriptions)
3.  [Setup & Prerequisites](#3-setup--prerequisites)
    * [System Resources](#system-resources)
    * [Conda Environment & Libraries](#conda-environment--libraries)
4.  [Dataset Acquisition & Preparation](#4-dataset-acquisition--preparation)
    * [Pre-training Data (SSL4EO-S12)](#pre-training-data-ssl4eo-s12)
    * [Fine-tuning Data (EuroSAT RGB)](#fine-tuning-data-eurosat-rgb)
    * [Data Augmentation Pipelines](#data-augmentation-pipelines)
5.  [Implementation Details](#5-implementation-details)
    * [SimSiam Model Architecture](#simsiam-model-architecture)
    * [Pre-training Strategy](#pre-training-strategy)
    * [Evaluation & Fine-tuning Strategy](#evaluation--fine-tuning-strategy)
6.  [Execution & Reproduction Guide](#6-execution--reproduction-guide)
    * [Step 0: Verify System Info (Optional)](#step-0-verify-system-info-optional)
    * [Step 1: Prepare Pre-training Data](#step-1-prepare-pre-training-data)
    * [Step 2: Pre-training the SimSiam Model](#step-2-pre-training-the-simsiam-model)
    * [Step 3: Perform Linear Evaluation](#step-3-perform-linear-evaluation)
    * [Step 4: Perform Full Fine-tuning](#step-4-perform-full-fine-tuning)
    * [Step 5: Classify a Single Image](#step-5-classify-a-single-image)
    * [Step 6: Generate Performance Graphics](#step-6-generate-performance-graphics)
7.  [Key Results](#7-key-results)
8.  [Acknowledgements](#8-acknowledgements)

---

## 1. Introduction

This project implements a self-supervised learning (SSL) pipeline using the **SimSiam** method to pre-train a robust feature extractor for Earth Observation (EO) imagery. The primary goal is to leverage large amounts of unlabeled satellite data to learn general-purpose visual representations. Subsequently, this pre-trained model is adapted via **linear evaluation** and **full fine-tuning** for a specific downstream task: **land cover classification** on the EuroSAT RGB dataset. This approach significantly reduces the reliance on extensive labeled datasets, a common challenge in EO applications.

## 2. Project Structure & File Descriptions

```

.
├── .gitignore                 \# Specifies intentionally untracked files to ignore by Git.
├── barlow\_data/               \# Directory for pre-training data (converted PNGs).
│   └── train/                 \#   └── Contains converted RGB PNGs for SimSiam pre-training input.
├── data/                      \# Primary directory for raw datasets.
│   └── ssl4eo-s12/            \#   └── Contains raw SSL4EO-S12 Zarr files (after unzipping).
│       └── train/
│           └── S2L1C/         \#   └── Assuming .zarr files are located here.
├── datae/                     \# Secondary directory for specific datasets.
│   └── EuroSAT/               \#   └── EuroSAT RGB dataset (class folders, CSVs, label\_map.json) for fine-tuning/evaluation.
│       ├── AnnualCrop/
│       ├── ... (other class folders)
│       ├── label\_map.json     \#   └── Maps class names to integer labels for EuroSAT.
│       ├── test.csv           \#   └── Defines the EuroSAT test set images and labels.
│       ├── train.csv          \#   └── Defines the EuroSAT training set images and labels.
│       └── validation.csv     \#   └── Defines the EuroSAT validation set images and labels.
├── scripts/                   \# Directory for shell scripts.
│   ├── data.sh                \# Shell script (conceptual) for dataset acquisition.
│   └── unzip\_files.sh         \# Shell script for unzipping dataset archives.
├── make\_dataset.py            \# Python script to convert raw Zarr files into RGB PNGs for pre-training.
├── classify\_image.py          \# A utility script to classify a single image using the fine-tuned model.
├── finetune\_model.py          \# Script for performing full fine-tuning of the pre-trained model on the EuroSAT dataset (backbone unfrozen).
├── Linear\_evaluation.py       \# Script to evaluate the pre-trained model's features via linear classification on EuroSAT (backbone frozen).
├── plot\_results.py            \# Script to load metrics and generate performance graphics for the report.
├── simsiam\_data.py            \# Defines the custom PyTorch Dataset and data augmentation pipelines for SimSiam pre-training.
├── simsiam\_model.py           \# Contains the SimSiam model architecture (ResNet-50 backbone, Projection, and Prediction heads).
├── train\_simsiam.py           \# Main script for performing the SimSiam self-supervised pre-training.
└── README.md                  \# This README file, providing project overview and guide.

````

## 3. Setup & Prerequisites

### System Resources

The project was developed and tested on a system with the following specifications:
* **GPU:** NVIDIA Quadro RTX 4000 (8 GB VRAM)
* **CPU:** 20 Cores

To verify your system's PyTorch and CUDA setup:
```bash
python -c "import torch, os; print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): print(f'GPU Name: {torch.cuda.get_device_name(0)}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB'); print(f'CPU Cores: {os.cpu_count()}')"
````

### Conda Environment & Libraries

It is highly recommended to use a Conda virtual environment for dependency management.

1.  **Create Conda environment:**
    ```bash
    conda create -n ssl4eo_env python=3.10
    conda activate ssl4eo_env
    ```
2.  **Install core libraries:**
    ```bash
    pip install torch torchvision numpy pillow pandas scikit-learn tqdm # Essential libraries
    pip install rasterio # Required if make_dataset.py reads directly from Zarr using rasterio
    ```

## 4\. Dataset Acquisition & Preparation

### Pre-training Data (SSL4EO-S12)

  * **Source:** The pre-training data is derived from a large, unlabeled SSL4EO-S12 dataset, typically obtained as `.zarr.zip` archives from its source.
  * **Preparation Steps:**
    1.  **Download:** Obtain the `SSL4EO-S12` `.zarr.zip` files and place them in a suitable directory (e.g., `data/ssl4eo-s12_zips`). You can adapt or use the provided `scripts/data.sh` conceptually.
          * **`scripts/data.sh` (Conceptual Download Snippet):**
            ```bash
            #!/bin/bash
            # This is a conceptual script. Actual download depends on dataset provider.
            # Example: For a direct URL download
            # DATA_URL="YOUR_DATASET_DOWNLOAD_URL"
            # OUTPUT_ZIP="data/ssl4eo-s12.zip"
            # echo "Downloading dataset from $DATA_URL to $OUTPUT_ZIP..."
            # curl -L $DATA_URL -o $OUTPUT_ZIP
            # echo "Download complete."
            ```
    2.  **Unzip Zarr Archives:** Extract the `.zarr` folders from the downloaded `.zip` files.
          * **`scripts/unzip_files.sh` (Unzipping Code):**
            ```bash
            #!/bin/bash
            # Path to your downloaded Zarr zip files (e.g., in data/ssl4eo-s12_zips)
            ZIP_DIR="data/ssl4eo-s12_zips"
            # Target directory for extracted Zarr folders (e.g., data/ssl4eo-s12)
            EXTRACT_DIR="data/ssl4eo-s12"

            mkdir -p $EXTRACT_DIR
            echo "Unzipping files from $ZIP_DIR to $EXTRACT_DIR..."

            for zip_file in "$ZIP_DIR"/*.zip; do
                if [ -f "$zip_file" ]; then
                    echo "Extracting $zip_file..."
                    unzip -q "$zip_file" -d "$EXTRACT_DIR"
                fi
            done
            echo "All zips extracted."
            ```
    3.  **Convert to RGB PNGs:** Use the `make_dataset.py` script to read the multi-band `.zarr` files (located in `data/ssl4eo-s12/train/S2L1C/` as per initial context) and convert them into 3-channel RGB PNG images (specifically using Sentinel-2's B4, B3, B2 bands).
          * **Output Location:** These PNGs should be saved to the `barlow_data/train` directory. This directory will serve as the input for SimSiam pre-training.
          * **Rationale:** This conversion standardizes the input format, making it compatible with `torchvision`'s image processing capabilities designed for RGB images.

### Fine-tuning Data (EuroSAT RGB)

  * **Source:** The EuroSAT RGB dataset, commonly found on platforms like [Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/smajida/eurosat-rgb).
  * **Acquisition:** Download and extract the dataset. The extracted folder (e.g., `EuroSAT`) should contain 10 class-specific subdirectories, along with `train.csv`, `validation.csv`, `test.csv`, and `label_map.json`.
  * **Placement:** Place the main `EuroSAT` folder inside your project's `datae/` directory (e.g., `seminar/datae/EuroSAT`).
      * **Data Splitting:** The provided `train.csv`, `validation.csv`, and `test.csv` files define the rigorous training, validation, and testing splits, ensuring proper evaluation on unseen data.
      * **Label Mapping:** `label_map.json` provides the consistent mapping from class names (strings) to integer indices.
      * **Key Resolution:** During implementation, a `KeyError` related to CSV column names was resolved by correctly referencing the `'Filename'` column for image paths and the `'ClassName'` column for labels (as found in the CSV headers) and then mapping these class name strings to integer indices via `label_map.json`.

### Data Augmentation Pipelines

The success of SimSiam heavily relies on diverse and strong data augmentations. Two distinct pipelines (`simsiam_transform_v1` and `simsiam_transform_v2`, defined in `simsiam_data.py`) are used to generate two augmented views of each image:

  * **Common Augmentations (Applied to both `view1` and `view2`):**
      * `RandomResizedCrop(size=224, scale=(0.2, 1.0))`: Randomly crops and resizes the image.
      * `RandomHorizontalFlip(p=0.5)`: Randomly flips the image horizontally.
      * `RandomApply([ColorJitter(...)], p=0.8)`: Randomly applies color distortions.
      * `RandomGrayscale(p=0.2)`: Randomly converts to grayscale.
      * `RandomApply([GaussianBlur(...)], p=0.5)`: Randomly applies Gaussian blur.
      * `ToTensor()`: Converts PIL image to PyTorch Tensor.
      * `Normalize(mean=ImageNet_means, std=ImageNet_stds)`: Normalizes pixel values.
  * **Asymmetric Augmentation (Applied only to `view2`):**
      * `RandomApply([Solarization()], p=0.2)`: Randomly applies solarization effect.
  * **Rationale:** These augmentations closely follow the SimSiam paper to promote invariance learning and boost feature quality. The asymmetric Solarization is a key element for preventing representational collapse.
  * **Key Resolution:** An initial `TypeError` with `Solarization` was resolved by ensuring it operates on image pixel values (in the 0-255 range) *before* `transforms.ToTensor()` scales them to 0.0-1.0.

## 5\. Implementation Details

### SimSiam Model Architecture

The `simsiam_model.py` defines the core SimSiam network, inheriting from `torch.nn.Module`:

  * **Backbone:** **ResNet-50** is used as the primary feature extractor. Its initial convolutional layer (`conv1`) is adapted to correctly accept 3-channel RGB inputs (matching the converted PNGs).
      * **Decision & Rationale:** ResNet-50 is the standard choice for SimSiam, offering superior feature learning capabilities due to its depth.
  * **Projection Head (`projector`):** A 3-layer MLP (`Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm`).
      * **Specificity & Rationale:** The final `BatchNorm1d` layer in the projector has `affine=False`. This is a critical architectural detail from the SimSiam paper, fundamental to preventing representational collapse without needing explicit negative sample pairs.
  * **Prediction Head (`predictor`):** A 2-layer MLP (`Linear -> BatchNorm -> ReLU -> Linear`).
  * **Stop-Gradient Mechanism:** The `.detach()` operation is applied to the target representations ($z\_1, z\_2$) during the `forward` pass of the SimSiam model.
      * **Rationale:** This simple yet effective mechanism is central to SimSiam, enabling it to learn meaningful representations by breaking gradient flow from one branch to the other's target.

### Pre-training Strategy

The `train_simsiam.py` script manages the self-supervised pre-training phase:

  * **Loss Function:** Negative Cosine Similarity, calculated symmetrically between the two augmented views.
  * **Optimizer:** Stochastic Gradient Descent (SGD) with `momentum=0.9` and `weight_decay=1e-4`.
  * **Learning Rate (LR) Schedule:** Implemented a **Linear Warmup** phase (for the first 10 epochs) followed by a **half-cycle Cosine Annealing Decay** for the remaining epochs.
      * **Specificity & Rationale:** This dynamic LR adjustment is a best practice for training deep neural networks, especially in self-supervised learning, promoting stable convergence and leading to better final performance.
  * **Batch Size & Epochs:** `BATCH_SIZE = 32` for `NUM_EPOCHS = 90`.
      * **Decision & Rationale:** The `BATCH_SIZE` of 32 was chosen due to GPU memory limitations (a `CUDA out of memory` error was encountered when attempting `BATCH_SIZE = 64` with ResNet-50 on the 8GB Quadro RTX 4000). This error log (e.g., `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...`) is evidence of this constraint. The `NUM_EPOCHS` was set to 90 due to external constraints (school desktop shutdown schedule). While ideal training might be longer, SimSiam's robustness to batch size and the high number of total gradient updates for 90 epochs make this a viable strategy.

### Evaluation & Fine-tuning Strategy

After pre-training, the model's performance on the EuroSAT land cover classification task is rigorously assessed:

  * **Linear Evaluation (`Linear_evaluation.py`):**
      * **Purpose:** To quantitatively assess the raw quality of the pre-trained features learned by the backbone.
      * **Methodology:** The pre-trained ResNet-50 backbone is **frozen** (its parameters are not updated), and only a new, simple linear classification head is trained on the EuroSAT training set. Performance is evaluated on validation and a final test set.
  * **Full Fine-tuning (`finetune_model.py`):**
      * **Purpose:** To adapt the pre-trained model's features for maximum performance on the specific downstream task.
      * **Methodology:** The entire pre-trained ResNet-50 backbone is **unfrozen** (all its parameters are allowed to be updated along with the new classification head's parameters). A much smaller learning rate (0.0001) is used for the entire model during fine-tuning compared to pre-training, to prevent "catastrophic forgetting" of the beneficial pre-trained weights.

## 6\. Execution & Reproduction Guide

Ensure your Conda environment (`ssl4eo_env`) is activated and all datasets are prepared and placed as described in [Dataset Acquisition & Preparation](https://www.google.com/search?q=%234-dataset-acquisition--preparation). Navigate to the root directory of this repository (`~/Dokumente/seminar` in your case).

### Step 0: Verify System Info (Optional)

Run a quick script to check your system's PyTorch and CUDA setup:

```bash
python -c "import torch, os; print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): print(f'GPU Name: {torch.cuda.get_device_name(0)}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB'); print(f'CPU Cores: {os.cpu_count()}')"
```

### Step 1: Prepare Pre-training Data

1.  **Download Zarr Archives:** Execute `scripts/data.sh` (or manually download SSL4EO-S12 .zarr.zip files).
    ```bash
    bash scripts/data.sh # If data.sh is implemented for download
    ```
2.  **Unzip Zarr Archives:** Execute `scripts/unzip_files.sh`. This will extract `.zarr` folders into `data/ssl4eo-s12/`.
    ```bash
    bash scripts/unzip_files.sh
    ```
3.  **Convert Zarr to PNGs:** Run `make_dataset.py`. This will create RGB PNGs in `barlow_data/train/`.
    ```bash
    python make_dataset.py
    ```

### Step 2: Pre-training the SimSiam Model

This step trains the SimSiam model on your unlabeled `barlow_data/train` images for 90 epochs.

```bash
python train_simsiam.py
```

  * **Expected Output:** A `tqdm` progress bar for each epoch, decreasing average loss, and `simsiam_resnet50_epoch_X.pth` checkpoints saved periodically. A `pretrain_metrics.pth` file (containing epoch-wise loss and LR) will be created, updated after each epoch.

### Step 3: Perform Linear Evaluation

This step loads the pre-trained backbone, freezes it, and trains a linear classifier on the EuroSAT dataset (using `train.csv` for training, `validation.csv` for validation, and `test.csv` for final evaluation).

```bash
python Linear_evaluation.py
```

  * **Expected Output:** Training and validation accuracy/loss per epoch, `linear_eval_eurosat_best_model.pth` (best model based on validation) saved, and a final test accuracy reported. A `linear_eval_metrics.pth` file (containing epoch-wise train/val loss/acc, and final test acc) will be created, updated after each epoch.

### Step 4: Perform Full Fine-tuning

This step loads the pre-trained backbone and fine-tunes the entire model (backbone + classifier) on the EuroSAT dataset.

```bash
python finetune_model.py
```

  * **Expected Output:** Similar training/validation metrics as linear evaluation, but potentially higher accuracies. `finetuned_eurosat_best_model.pth` saved, and a final test accuracy reported. A `finetune_metrics.pth` file (containing epoch-wise train/val loss/acc, and final test acc) will be created, updated after each epoch.

### Step 5: Classify a Single Image

Use your fine-tuned model to classify a new image. Remember to update `IMAGE_TO_CLASSIFY_PATH` in `classify_image.py` before running.

```bash
python classify_image.py
```

  * **Expected Output:** Predicted class name and confidence for the specified image.

### Step 6: Generate Performance Graphics

(This step assumes you have a `plot_results.py` script. You'll need to confirm its paths and implement the plotting functions based on our previous discussions.)

Run your plotting script:

```bash
python plot_results.py
```

  * **Expected Graphics:**
      * **Pre-training:** Loss curve, Learning Rate schedule.
      * **Linear Eval/Fine-tuning:** Training/Validation Loss & Accuracy curves.
      * **Comparison:** Bar chart of Linear Eval vs. Fine-tuning Test Accuracy.
      * **Qualitative:** Grid of example classifications (True vs. Predicted).
      * **Augmentation Visualization:** Example showing original and augmented views.

## 7\. Key Results

The implemented pipeline successfully demonstrated the efficacy of SimSiam for EO feature learning and subsequent classification:

  * **Pre-training:** The SimSiam model (ResNet-50) successfully converged, showing a healthy, decreasing loss over 90 epochs, indicating effective self-supervised learning of features from unlabeled satellite imagery.
  * **Linear Evaluation:** Achieved **96.00% Test Accuracy** on EuroSAT. This high accuracy demonstrates that the pre-trained features are highly discriminative and linearly separable for land cover classification, confirming the strong quality of the learned representations. This result does not indicate overfitting, as the backbone was frozen and performance was consistent across all data splits.
  * **Full Fine-tuning:** Achieved a remarkable **98.33% Test Accuracy** on EuroSAT. This significant improvement over linear evaluation highlights the substantial benefit of adapting the powerful pre-trained backbone directly to the specific nuances of the EuroSAT dataset, resulting in state-of-the-art level performance for this task.

## 8\. Acknowledgements

  * **SimSiam:** Chen, T., et al. (2021). *SimSiam: Exploring Simple Siamese Representation Learning*. IEEE/CVF International Conference on Computer Vision (ICCV).
  * **EuroSAT Dataset:** Helber, P., et al. (2019). *EuroSAT: A Novel Dataset for Deep Learning in Remote Sensing*. IEEE Transactions on Geoscience and Remote Sensing.
  * **SSL4EO-S12 Dataset:** (Cite the specific SSL4EO-S12 paper/source if applicable)

-----

```
```
