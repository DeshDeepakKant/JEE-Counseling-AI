# Results and Discussion: Performance on Datasets

SwarmFusionDNN was tested across three vastly different benchmark datasets, showcasing extreme adaptability and overall diagnostic accuracy.

## 1. Br35H Dataset (Binary Classification)

The model aimed to accurately identify whether MRI images contained a tumor or reflected healthy tissue.
- **Accuracy:** 99.66% 
- **Error Rate:** 0.34%
- **AUC (Area Under Curve):** 0.9998
- **Base Models Context:** MobileNet achieved roughly 99.33%, while MobileNetV2 dipped to 94.66%. The PSO ensemble effortlessly pulled the accuracy up.
- **Misclassifications:** Under optimal fusion, SwarmFusionDNN simply misclassified 2 instances out of 600 validation queries. 

## 2. Figshare Dataset (Multi-Class Classification)

To evaluate multi-class capability, the model was tasked to classify Gliomas, Meningiomas, and Pituitary tumors.
- **Accuracy:** 96.14% 
- **Class-wise AUC:** Pituitary tumor discrimination peaked effectively at 99.74% via the SwarmFusionDNN classifier.
- **Base Models Context:** DenseNet121 operated impressively effectively here (94.85%), but again, the SwarmFusionDNN optimized weighting achieved the highest overall synergy.
- **Misclassifications:** Misclassed an incredibly low 24 out of 622 testing images. Meningiomas were occasionally misdiagnosed as pituitary tumors due to similar anatomy features.

## 3. Bangladesh Brain Cancer Dataset (Multi-Class Classification)

This realistic dataset features 6,056 resized test queries. SwarmFusionDNN continued to rank highest across independent models. 
- **Accuracy:** 98.76% 
- **AUC:** 0.9991
- **Base Models Context:** MobileNet and MobileNetV2 matched predictions precisely here (97.61%), but SwarmFusionDNN pushed error metrics dramatically further downwards.
- **Misclassifications:** 15 misclassifications out of 1,212 samples. 

Overall, tests indicate the SwarmIntelligence Optimization and Weight averaging actively prevent overfitting, relying heavily on the collective strengths of its independent CNNs.
