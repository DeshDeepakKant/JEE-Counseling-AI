# Methodology: Base Models and ConvMixer

## Deep Feature Extraction via Pre-trained CNNs

To extract deep features from the MRI images, the extended **SwarmFusionDNN** employs **six** pre-trained CNN architectures. The original four models from the paper have been augmented with two additional architectures — **InceptionV3** and **ResNet50** — to further strengthen the ensemble's feature diversity and classification robustness. Each model processes 224×224 MRI images with its backbone frozen (ImageNet weights) and a ConvMixer block appended.

### 1. MobileNet
- **Accuracy:** 91.32% (baseline)
- **Features:** A lightweight CNN utilizing depthwise separable convolutions to reduce parameters and computational complexity. Well-suited for clinical settings.

### 2. DenseNet121
- **Accuracy:** 87.90% (baseline)
- **Features:** Uses dense connectivity meaning it promotes feature reuse across layers. Its 121 layers enable extraction of deeper and more intricate patterns from brain scans.

### 3. MobileNetV2
- **Accuracy:** 84.10% (baseline)
- **Features:** Introduces inverted residual blocks and linear bottlenecks. Known for its small size and lightning-fast inference time.

### 4. Xception
- **Accuracy:** 81.70% (baseline)
- **Features:** Rooted in the inception module, relying entirely on depthwise separable convolutions to capture complex patterns across spatial dimensions and feature channels.

### 5. InceptionV3 *(Extended — New Model)*
- **Features:** An advanced evolution of the original Inception architecture by Google. Uses factorised convolutions (e.g. 1×7 and 7×1 instead of 7×7) to dramatically reduce parameters while maintaining a rich, multi-scale receptive field. The deep inception modules are particularly effective at capturing subtle textural differences between tumour types.
- **Why added:** Complements the existing ensemble by capturing multi-scale features at different resolutions simultaneously, which the original four models do not fully exploit.

### 6. ResNet50 *(Extended — New Model)*
- **Features:** A 50-layer residual network that uses **skip connections** to allow gradients to flow directly across many layers, effectively preventing the vanishing-gradient problem. Its deep residual blocks learn highly discriminative hierarchical features that are well-suited to identifying complex tumour morphology.
- **Why added:** Residual connections give ResNet50 a fundamentally different inductive bias compared to the depthwise-separable or dense-connectivity approaches of the original four models, increasing ensemble diversity and robustness.

---

## ConvMixer Block Integration

Pre-trained networks alone are often susceptible to redundancy and overlapping spatial features. To counteract this, standard models are modified by appending a **ConvMixer block** to enhance the extracted features.

### Architecture of ConvMixer
The ConvMixer block employs a combination of **depthwise and pointwise convolutions**:
1. **Depthwise Convolution:** Captures detailed **spatial** correlations.
2. **Pointwise Convolution:** Evaluates and shapes interactions between **feature channels**.

By mixing these features, ConvMixer reduces feature overlap (redundancy), expands the model's receptive field, and ensures robust generalization. A **GeLU** activation function handles non-linearities, and **batch normalization** stabilizes training. 

By applying **Alpha Dropout (rate=0.5)**, SwarmFusionDNN ensures models do not overly rely on individual neurons (overfitting evasion), passing the finalized, rich feature representations directly into the classification head.
