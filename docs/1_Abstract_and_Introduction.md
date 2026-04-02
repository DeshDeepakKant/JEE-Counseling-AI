# Abstract and Introduction

## Abstract

Accurate identification of brain tumors is crucial for early diagnosis and the development of effective treatment strategies. Many existing deep learning methods rely on simple ensemble approaches without utilizing advanced optimization techniques. 

This research proposes **SwarmFusionDNN**, a novel ensemble approach for brain tumor detection using a weighted average ensemble. 
- It incorporates **ConvMixer blocks** into popular pre-trained models (MobileNet, DenseNet121, MobileNetV2, Xception) to enhance feature extraction (spatial and channel-wise information) from MRI scans.
- To maximize ensemble performance, a **Particle Swarm Optimization (PSO)** algorithm minimizes the classifier error rate to find the best possible weights assigned to each base model.

Tested on three widely recognized benchmark MRI datasets (Br35H, Figshare, and Bangladesh Brain Cancer), SwarmFusionDNN outperformed individual models, achieving:
- **99.66%** accuracy on the Br35H dataset.
- **96.14%** accuracy on the Figshare dataset.
- **98.76%** accuracy on the Bangladesh Brain Cancer dataset.

Additionally, **Grad-CAM visualization** confirms that the model accurately focuses on tumor regions, offering improved transparency and interpretability over traditional ensemble methods.

---

## Introduction

Brain tumors are complex neoplasms, broadly classified into meningioma, glioma, and pituitary tumors. Early detection and classification via medical imaging like MRIs are critical for patient prognosis. Artificial Intelligence, specifically Deep Learning (DL) and Transfer Learning (TL), has revolutionized this field by learning intricate patterns in large datasets of annotated brain scans.

However, recognizing optimal weights for different models within an ensemble can be a pivotal factor for achieving top performance:
- Existing ensemble systems often use static or experimentally determined weights, which are time-consuming and inefficient.
- To address this, the authors introduce a **Swarm Intelligence-based approach**.

**Key Contributions:**
1. **Architecture:** Base models (MobileNet, DenseNet121, MobileNetV2, Xception) are enhanced with ConvMixer blocks to capture intricate spatial/channel-wise information.
2. **Optimization:** PSO algorithm is used to tune the weights of these base models in the ensemble, directly minimizing the classification error.
3. **Rigorous Evaluation:** Testing across three different MRI datasets demonstrates the practical applicability and superior accuracy of SwarmFusionDNN.
4. **Interpretability:** Grad-CAMs confirm localization of tumors within the MRI scans, demonstrating reliable feature detection.
