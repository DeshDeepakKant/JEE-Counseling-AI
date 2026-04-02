# Results and Discussion: Experimental Setup

All models were evaluated using the standard **80-20 split** (80% training, 20% testing).

## Evaluation Metrics

To thoroughly analyze performance across multiple MRI diagnostic levels, the study primarily measured:

- **Accuracy (ACCU):** Indicates how well the models correctly classified both True Positives (tumor) and True Negatives (normal).
- **Sensitivity (Recall/SENS):** Signifies the model's capacity to correctly recognize ground-truth tumors (vital metric to minimize misdiagnosis). 
- **Precision (PREC):** Indicates how many images identified as tumors were genuinely tumors (assesses overprediction rates).
- **F1-Score:** Harmonic evaluation of Precision and Recall.
- **Specificity (SPEC):** Correct identification of non-tumors (reduces false alarms).

## Implementation Details

All tests and architectures relied upon TensorFlow and Keras implementations. Model training required roughly **20 epochs**, executing on:

- **Processor:** Ryzen 5
- **GPU:** NVIDIA MX450
- **RAM:** 16 GB 

### Hyperparameters
- **Optimizer:** Adam
- **Learning Rate:** Dynamically adjusts dynamically by a factor of 0.5 when performance plateaus (`ReduceLROnPlateau`). Initial rate is $1e-3$. 
- **Batch Size:** 64
- **Loss Function:** Cross-entropy.
- **Regularization:** Used early stopping callbacks (Patience: 12) coupled with an Alpha Dropout rate of 0.5.

The open-source code and implementation detail repository is hosted at [GitHub](https://github.com/sohaibasif21/SwarmFusionDNN).
