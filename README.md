You mean you want the **entire README content** in a single, plain text block that you can copy and paste directly into GitHub's web interface if you were editing the `README.md` file there?

Okay, here is the complete, raw Markdown content for your `README.md` file. You can copy everything below this line and paste it into the `README.md` file editor on GitHub, or into a local `README.md` file in your project.

```markdown
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
    * [Step 1: Pre-training the SimSiam Model](#step-1-pre-training-the-simsiam-model)
    * [Step 2: Perform Linear Evaluation](#step-2-perform-linear-evaluation)
    * [Step 3: Perform Full Fine-tuning](#step-3-perform-full-fine-tuning)
    * [Step 4: Classify a Single Image](#step-4-classify-a-single-image)
    * [Step 5: Generate Performance Graphics](#step-5-generate-performance-graphics)
7.  [Key Results](#7-key-results)
8.  [Acknowledgements](#8-acknowledgements)

---

## 1. Introduction

This project implements a self-supervised learning (SSL) pipeline using the **SimSiam** method to pre-train a robust feature extractor for Earth Observation (EO) imagery. The primary goal is to leverage large amounts of unlabeled satellite data to learn general-purpose visual representations. Subsequently, this pre-trained model is adapted via **linear evaluation** and **full fine-tuning** for a specific downstream task: **land cover classification** on the EuroSAT RGB dataset. This approach significantly reduces the reliance on extensive labeled datasets, a common challenge in EO applications.

## 2. Project Structure & File Descriptions

```

.
├── barlow\_data/               \# Directory for pre-training data (converted PNGs)
│   └── train/                 \#   └── Contains converted RGB PNGs for SimSiam pre-training input.
├── data/                      \# Directory for all other datasets.
│   └── EuroSAT/               \#   └── EuroSAT RGB dataset (class folders, CSVs, label\_map.json) for fine-tuning/evaluation.
│       ├── AnnualCrop/
│       ├── ... (other class folders)
│       ├── label\_map.json     \#   └── Maps class names to integer labels for EuroSAT.
│       ├── test.csv           \#   └── Defines the EuroSAT test set images and labels.
│       ├── train.csv          \#   └── Defines the EuroSAT training set images and labels.
│       └── validation.csv     \#   └── Defines the EuroSAT validation set images and labels.
├── simsiam\_data.py            \# Defines the custom PyTorch Dataset and data augmentation pipelines for SimSiam pre-training.
├── simsiam\_model.py           \# Contains the SimSiam model architecture (ResNet-50 backbone, Projection, and Prediction heads).
├── train\_simsiam.py           \# Main script for performing the SimSiam self-supervised pre-training.
├── Linear\_evaluation.py       \# Script to evaluate the pre-trained model's features via linear classification on EuroSAT (backbone frozen).
├── finetune\_model.py          \# Script for performing full fine-tuning of the pre-trained model on the EuroSAT dataset (backbone unfrozen).
├── classify\_image.py          \# A utility script to classify a single image using the fine-tuned model.
├── plot\_results.py            \# (To be created) Script to load metrics and generate performance graphics for the report.
├── .gitignore                 \# Specifies intentionally untracked files to ignore by Git.
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

Create Conda environment:

```bash
conda create -n ssl4eo_env python=3.10
conda activate ssl4eo_env
```

Install core libraries:

```bash
pip install torch torchvision numpy pillow pandas scikit-learn tqdm # Essential libraries
pip install rasterio # Required if adapting zarr_to_images.py for direct Zarr reading
```

## 4\. Dataset Acquisition & Preparation

### Pre-training Data (SSL4EO-S12)

  * **Source:** The pre-training data is derived from a large, unlabeled SSL4EO-S12 dataset, typically obtained as `.zarr.zip` archives from its source.

  * **Preparation Steps:**

    1.  **Download:** Obtain the SSL4EO-S12 `.zarr.zip` files and place them in a `data/ssl4eo-s12_zips` directory (or similar).
        ```python
        # Placeholder for large dataset download. Actual download depends on the dataset provider.
        # For very large files, use streaming downloads or specific dataset APIs.
        # Example for a direct URL:
        # import requests
        # url = "YOUR_DATASET_DOWNLOAD_URL"
        # output_path = "data/ssl4eo-s12.zip"
        # print(f"Downloading {url} to {output_path}...")
        # # with requests.get(url, stream=True) as r: ... # Use streaming for large files
        # # print("Download complete.")
        ```
    2.  **Unzip Zarr Archives:** Extract the `.zarr` folders from the downloaded `.zip` files.
        ```python
        import zipfile
        import os

        zip_file_path = "data/ssl4eo-s12.zip" # Path to your Zarr zip file
        extract_to_path = "data/ssl4eo-s12_extracted" # Target directory for Zarr folders

        print(f"Unzipping {zip_file_path} to {extract_to_path}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print("Unzipping complete.")
        ```
    3.  **Convert to RGB PNGs:** Use a custom script (e.g., `zarr_to_images.py`, not provided in this repo but used during the process) to read the multi-band `.zarr` files (e.g., from `data/ssl4eo-s12_extracted`) and convert them into 3-channel RGB PNG images (specifically using Sentinel-2's B4, B3, B2 bands).

    <!-- end list -->

      * **Output Location:** These PNGs should be saved to the `barlow_data/train` directory. This directory will serve as the input for SimSiam pre-training.
      * **Rationale:** This conversion standardizes the input format, making it compatible with `torchvision`'s image processing capabilities designed for RGB images.

### Fine-tuning Data (EuroSAT RGB)

  * **Source:** The EuroSAT RGB dataset, commonly found on platforms like Kaggle.
  * **Acquisition:** Download and extract the dataset. The extracted folder (e.g., `EuroSAT`) should contain 10 class-specific subdirectories, along with `train.csv`, `validation.csv`, `test.csv`, and `label_map.json`.
  * **Placement:** Place the main `EuroSAT` folder inside your project's `data/` directory (e.g., `seminar/data/EuroSAT`).
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

  * **Key Resolution:** An initial `TypeError` with Solarization was resolved by ensuring it operates on image pixel values (in the 0-255 range) before `transforms.ToTensor()` scales them to 0.0-1.0.

## 5\. Implementation Details

### SimSiam Model Architecture

The `simsiam_model.py` defines the core SimSiam network, inheriting from `torch.nn.Module`:

  * **Backbone:** ResNet-50 is used as the primary feature extractor. Its initial convolutional layer (`conv1`) is adapted to correctly accept 3-channel RGB inputs (matching the converted PNGs).
      * **Decision & Rationale:** ResNet-50 is the standard choice for SimSiam, offering superior feature learning capabilities due to its depth.
  * **Projection Head (`projector`):** A 3-layer MLP (`Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm`).
      * **Specificity & Rationale:** The final `BatchNorm1d` layer in the projector has `affine=False`. This is a critical architectural detail from the SimSiam paper, fundamental to preventing representational collapse without needing explicit negative sample pairs.
  * **Prediction Head (`predictor`):** A 2-layer MLP (`Linear -> BatchNorm -> ReLU -> Linear`).
  * **Stop-Gradient Mechanism:** The `.detach()` operation is applied to the target representations ($z\_1, z\_2$) during the forward pass of the SimSiam model.
      * **Rationale:** This simple yet effective mechanism is central to SimSiam, enabling it to learn meaningful representations by breaking gradient flow from one branch to the other's target.

### Pre-training Strategy

The `train_simsiam.py` script manages the self-supervised pre-training phase:

  * **Loss Function:** Negative Cosine Similarity, calculated symmetrically between the two augmented views.
  * **Optimizer:** Stochastic Gradient Descent (SGD) with `momentum=0.9` and `weight_decay=1e-4`.
  * **Learning Rate (LR) Schedule:** Implemented a Linear Warmup phase (for the first 10 epochs) followed by a half-cycle Cosine Annealing Decay for the remaining epochs.
      * **Specificity & Rationale:** This dynamic LR adjustment is a best practice for training deep neural networks, especially in self-supervised learning, promoting stable convergence and leading to better final performance.
  * **Batch Size & Epochs:** `BATCH_SIZE = 32` for `NUM_EPOCHS = 200`.
      * **Decision & Rationale:** The `BATCH_SIZE` of 32 was chosen due to GPU memory limitations (a `CUDA out of memory` error was encountered when attempting `BATCH_SIZE = 64` with ResNet-50 on the 8GB Quadro RTX 4000). This error log (e.g., `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...`) is evidence of this constraint. The `NUM_EPOCHS` was increased to 200 to compensate for the smaller batch size, ensuring a high total number of gradient updates (161,600 steps) for effective learning, leveraging SimSiam's inherent robustness to batch size. This contrasts with the paper's typical 100 epochs/BS256 configuration which performs fewer total steps.

### Evaluation & Fine-tuning Strategy

After pre-training, the model's performance on the EuroSAT land cover classification task is rigorously assessed:

  * **Linear Evaluation (`Linear_evaluation.py`):**
      * **Purpose:** To quantitatively assess the raw quality of the pre-trained features learned by the backbone.
      * **Methodology:** The pre-trained ResNet-50 backbone is frozen (its parameters are not updated), and only a new, simple linear classification head is trained on the EuroSAT training set. Performance is evaluated on validation and a final test set.
  * **Full Fine-tuning (`finetune_model.py`):**
      * **Purpose:** To adapt the pre-trained model's features for maximum performance on the specific downstream task.
      * **Methodology:** The entire pre-trained ResNet-50 backbone is unfrozen (all its parameters are allowed to be updated along with the new classification head's parameters). A much smaller learning rate (0.0001) is used for the entire model during fine-tuning compared to pre-training, to prevent "catastrophic forgetting" of the beneficial pre-trained weights.

## 6\. Execution & Reproduction Guide

Ensure your Conda environment (`ssl4eo_env`) is activated and all datasets are prepared and placed as described in [Dataset Acquisition & Preparation](https://www.google.com/search?q=%234-dataset-acquisition--preparation). Navigate to the root directory of this repository (`~/Dokumente/seminar` in your case).

### Step 0: Verify System Info (Optional)

Run a quick script to check your system's PyTorch and CUDA setup:

```bash
python -c "import torch, os; print(f'CUDA available: {torch.cuda.is_is_available()}'); if torch.cuda.is_available(): print(f'GPU Name: {torch.cuda.get_device_name(0)}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB'); print(f'CPU Cores: {os.cpu_count()}')"
```

### Step 1: Pre-training the SimSiam Model

This step trains the SimSiam model on your unlabeled `barlow_data/train` images. This will take a significant amount of time (approx. 5.5 minutes per epoch for 200 epochs on the specified hardware).

```bash
python train_simsiam.py
```

  * **Expected Output:** A `tqdm` progress bar for each epoch, decreasing average loss, and `simsiam_resnet50_epoch_X.pth` checkpoints saved periodically. A `pretrain_metrics.pth` file (containing epoch-wise loss and LR) will be created upon completion.

### Step 2: Perform Linear Evaluation

This step loads the pre-trained backbone, freezes it, and trains a linear classifier on the EuroSAT dataset (using `train.csv` for training, `validation.csv` for validation, and `test.csv` for final evaluation).

```bash
python Linear_evaluation.py
```

  * **Expected Output:** Training and validation accuracy/loss per epoch, `linear_eval_eurosat_best_model.pth` (best model based on validation) saved, and a final test accuracy reported. A `linear_eval_metrics.pth` file (containing epoch-wise train/val loss/acc, and final test acc) will be created.

### Step 3: Perform Full Fine-tuning

This step loads the pre-trained backbone and fine-tunes the entire model (backbone + classifier) on the EuroSAT dataset.

```bash
python finetune_model.py
```

  * **Expected Output:** Similar training/validation metrics as linear evaluation, but potentially higher accuracies. `finetuned_eurosat_best_model.pth` saved, and a final test accuracy reported. A `finetune_metrics.pth` file (containing epoch-wise train/val loss/acc, and final test acc) will be created.

### Step 4: Classify a Single Image

Use your fine-tuned model to classify a new image. Remember to update `IMAGE_TO_CLASSIFY_PATH` in `classify_image.py` before running.

```bash
python classify_image.py
```

  * **Expected Output:** Predicted class name and confidence for the specified image.

### Step 5: Generate Performance Graphics

(This step assumes you have a `plot_results.py` script. You'll need to create it separately, combining the plotting snippets provided in our conversation, to load the generated `*.pth` metrics files and create visuals.)

Run your plotting script:

```bash
python plot_results.py
```

  * **Expected Graphics:**
      * **Pre-training:** Loss curve, Learning Rate schedule.
      * **Linear Eval/Fine-tuning:** Training/Validation Loss & Accuracy curves.
      * **Comparison:** Bar chart of Linear Eval vs. Fine-tuning Test Accuracy.
      * **Qualitative:** Grid of example classifications (True vs. Predicted).

## 7\. Key Results

The implemented pipeline successfully demonstrated the efficacy of SimSiam for EO feature learning and subsequent classification:

  * **Pre-training:** The SimSiam model (ResNet-50) successfully converged, showing a healthy, decreasing loss, indicating effective self-supervised learning of features from unlabeled satellite imagery.
  * **Linear Evaluation:** Achieved **95.81% Test Accuracy** on EuroSAT. This high accuracy demonstrates that the pre-trained features are highly discriminative and linearly separable for land cover classification, confirming the strong quality of the learned representations. This result does not indicate overfitting, as the backbone was frozen and performance was consistent across all data splits.
  * **Full Fine-tuning:** Achieved a remarkable **98.63% Test Accuracy** on EuroSAT. This significant improvement over linear evaluation highlights the substantial benefit of adapting the powerful pre-trained backbone directly to the specific nuances of the EuroSAT dataset, resulting in state-of-the-art level performance for this task.

## 8\. Acknowledgements

  * **SimSiam:** Chen, T., et al. (2021). SimSiam: Exploring Simple Siamese Representation Learning. IEEE/CVF International Conference on Computer Vision (ICCV).
  * **EuroSAT Dataset:** Helber, P., et al. (2019). EuroSAT: A Novel Dataset for Deep Learning in Remote Sensing. IEEE Transactions on Geoscience and Remote Sensing.
  * **SSL4EO-S12 Dataset:** (Cite the specific SSL4EO-S12 paper/source if applicable)

<!-- end list -->

```
```
