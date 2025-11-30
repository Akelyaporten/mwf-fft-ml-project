# Regression & FFT-based Damage Classification (MWF & CFRP)


1. **Metal Working Fluid (MWF) Regression**  
   Predicting the distance between a sensor and a drilling location from high-dimensional FFT features.

2. **CFRP Damage Classification from FFT Features**  
   Classifying different damage mechanisms in carbon fiber-reinforced polymers (CFRP) using deep learning.


> The original datasets are *not* included in this repository due to copyright and confidentiality.

---

## 1. Metal Working Fluid Regression

### Problem

Given:
- ~22,507 FFT coefficients per sample as features  
- Additional descriptors (segment, drilling direction, folder, file indices)  
- Target: **distance** between sensor and drilling location :contentReference[oaicite:7]{index=7}  

Goal:
- Learn a regression model that predicts the distance.
- Investigate:
  - Are all features needed?
  - Can dimensionality be reduced (PCA, feature selection)?
  - Which measurement series contribute most to model performance?

### Methods

- Standardization of features (`zscore`) based on training split.
- Optional preprocessing:
  - Principal Component Analysis (PCA) to reduce dimension while preserving ~95% variance. :contentReference[oaicite:8]{index=8}  
  - Variance thresholding and correlation filtering to remove near-constant and irrelevant features.
- Models:
  - **MLP (raw features)** – deep fully-connected network with dropout.
  - **MLP + PCA** – MLP on PCA-transformed features.
  - **CNN (1D)** – convolutional network on the FFT feature vector.
  - **Random Forest Regressor** as a classical machine learning baseline. :contentReference[oaicite:9]{index=9}  

### Results (Test set)

| Model              | RMSE  | MAE   | R²      |
|--------------------|-------|-------|---------|
| Random Forest      | 5.84  | 5.08  | 0.39    |
| MLP (raw features) | 7.88  | 6.00  | 0.01    |
| PCA + MLP          | 8.70  | 6.92  | -0.36   |
| CNN (raw features) | 8.37  | 7.17  | -0.12   |

*(Values from the project report; see `reports/Project_Work.pdf` for details.)* 

**Key observations:**

- Standardization was crucial for NN training stability.
- PCA + MLP achieved similar error to the CNN but with **much less compute**.
- With the given data size, **Random Forest clearly outperformed all neural networks**.

---

## 2. CFRP Damage Classification from FFT Features

### Problem

Dataset:
- 1,600 samples (FFT-based features of AE signals)
- 126 predictors: FFT coefficients between 10–500 kHz
- 5 classes:
  - Noise (no damage)
  - Delamination
  - Debonding
  - Matrix crack
  - Fiber breakage

Goal:
- Classify the damage mechanism from FFT features.
- Compare:
  1. **Two-stage approach**  
     - Stage 1: Noise vs Damage (binary)  
     - Stage 2: Damage type (4 classes)
  2. **Single-stage approach**  
     - Direct 5-class classification.

### Methods

- Custom feature standardization with a minimum standard deviation floor to avoid exploding values for near-constant features. :contentReference[oaicite:12]{index=12}  
- Feature analysis:
  - Mutual Information scores (Matlab `fscmrmr`) to estimate feature relevance.
  - Variance histograms to detect near-constant features. :contentReference[oaicite:13]{index=13}  
- Models:
  - 1D **CNN** (3 conv blocks + global average pooling + softmax).
  - Deep **MLP** (1024 → 512 → 256 → 128 → 64 → 32 → softmax).

### Results (Validation)

| Approach | Task                              | Accuracy (%) |
|----------|-----------------------------------|--------------|
| Two-stage | Step 1: Damage vs No damage     | 96.56        |
|          | Step 2: Damage type (4 classes)  | 88.96        |
|          | **Combined total accuracy**      | **91.25**    |
| Single-stage | 5-class classification       | **94.06**    |



**Key observations:**

- CNN converged quickly but generalized poorly in the second stage.
- MLP converged slower but was **more stable and accurate overall**.
- The **single-stage MLP** achieved the best overall performance with **simpler training** and roughly half the training time of the two-stage approach.

---


