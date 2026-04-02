# SwarmFusionDNN: Brain Tumor Detection

This repository contains the documentation for the research paper: **"Swarm intelligence optimization-based fusion of ConvMixer-enhanced deep neural networks for brain tumor detection"** by Sohaib Asif, Rongbiao Ying, and Enyu Wang.

## Table of Contents

1. [Abstract and Introduction](1_Abstract_and_Introduction.md)
2. **Methodology**
   - [Datasets](Methodology/1_Datasets.md)
   - [Base Models & ConvMixer](Methodology/2_Base_Models_and_ConvMixer.md)
   - [SwarmFusionDNN Architecture](Methodology/3_SwarmFusionDNN_Architecture.md)
   - [PSO Optimization](Methodology/4_PSO_Optimization.md)
3. **Results and Discussion**
   - [Experimental Setup](Results_and_Discussion/1_Experimental_Setup.md)
   - [Performance on Datasets](Results_and_Discussion/2_Performance_on_Datasets.md)
   - [Grad-CAM Visualization](Results_and_Discussion/3_Grad_CAM_Visualization.md)
   - [Comparison with Other Methods](Results_and_Discussion/4_Comparison_with_Other_Methods.md)
4. [Conclusion and Future Work](Conclusion_and_Future_Work.md)

## Overview

The research proposes **SwarmFusionDNN**, a novel ensemble approach for brain tumor detection. It enhances pre-trained deep neural networks by adding ConvMixer blocks, then fuses these models using a weighted average ensemble where the fusion weights are optimized by a Particle Swarm Optimization (PSO) algorithm to minimize the classifier error rate.

**This implementation extends the original paper from 4 to 6 base models:**

| # | Model | Type |
|---|---|---|
| 1 | MobileNet | Original (paper) |
| 2 | DenseNet121 | Original (paper) |
| 3 | MobileNetV2 | Original (paper) |
| 4 | Xception | Original (paper) |
| 5 | **InceptionV3** | **Extended (new)** |
| 6 | **ResNet50** | **Extended (new)** |

## Implementation Files

- **[SwarmFusionDNN_6Models.ipynb](../SwarmFusionDNN_6Models.ipynb)** — Jupyter notebook (recommended for step-by-step execution)
- **[SwarmFusionDNN_6Models.py](../SwarmFusionDNN_6Models.py)** — Standalone Python script
- **[requirements.txt](../requirements.txt)** — Package dependencies
