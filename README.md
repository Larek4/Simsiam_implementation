# SimSiam for Earth Observation: Self-Supervised Learning for Land Cover Classification

## Table of Contents
1.  [Introduction](#1-introduction)
2.  [Project Structure](#2-project-structure)
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

## 2. Project Structure
