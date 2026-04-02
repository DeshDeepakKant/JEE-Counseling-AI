# Methodology: Datasets

The study utilized three diverse brain MRI datasets to assess the generalizability and robustness of the proposed SwarmFusionDNN model.

## 1. BR35H Dataset
- **Source:** [Kaggle - Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
- **Total Images:** 3,000 MRI images.
- **Classes (Binary):** 
  - Tumor: 1,500 images (primarily glioma and meningioma)
  - Normal (Healthy): 1,500 images
- **Details:** Images generally consist of T1-weighted MRI sequences. The tumor sizes range from small localized patches to larger complexes.

## 2. Figshare Dataset
- **Source:** [Figshare - Brain Tumor Dataset](https://doi.org/10.6084/m9.figshare.1512427)
- **Total Images:** 3,064 MRI slices from 233 patients.
- **Classes (Multi-class):**
  - Glioma: 1,426 images
  - Meningioma: 708 images
  - Pituitary Tumors: 930 images
- **Details:** Acquired from a range of MRI modalities (T1-weighted, T2-weighted, contrast-enhanced). The dataset features significant diversity in terms of tumor stage, type, and size.

## 3. Bangladesh Brain Cancer MRI Dataset
- **Source:** [Mendeley Data - Brain Tumor MRI Dataset](https://doi.org/10.17632/mk56jw9rns.1)
- **Total Images:** 6,056 MRI images.
- **Classes (Multi-class):**
  - Glioma: 2,004 images
  - Meningioma: 2,004 images
  - Tumor (Generic): 2,048 images
- **Details:** Sourced from hospitals across Bangladesh, resizing initially to 512x512. Notable for real-world diversity containing various imaging modalities and patient demographics.

## Pre-processing & Data Augmentation

Data augmentation acts as a crucial step for preventing model overfitting, especially within medical imaging tasks:
- **Resizing:** All images standardized to **224 × 224 pixels**.
- **Augmentation Techniques:**
  - Horizontal and Vertical flips.
  - Shear range transformations (0.2).
  - Rotation (±20°).
- **Data Splitting:** Each dataset was split using an **80-20 ratio** (80% used for training, 20% reserved for testing/evaluation).
