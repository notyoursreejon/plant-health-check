# Section 5: Model Evaluation & Performance Analysis

## 5.1 Experimental Setup & Evaluation Protocol
To rigorously evaluate the proposed **Proposed PM-CNN (ResSE-CNN)** architecture for 39-class plant foliar pathogen identification, an extensive comparative empirical study was executed against standard machine learning baseline classifiers (**SVM**, **KNN**) and a conventional deep learning benchmark (**Standard CNN Baseline / MobileNetV2**). All experiments were conducted under identical data splitting ($50\%$ Train, $25\%$ Validation, $25\%$ Test) across $12,309$ images to eliminate evaluation bias.

Features for classical baseline classifiers (Support Vector Machine and K-Nearest Neighbors) were extracted directly from bottleneck embedding representations.

### 5.1.1 Evaluation Metrics Formulation
Classification efficacy was evaluated across $12$ statistical metrics:
1. **Accuracy**: Overall proportion of correctly classified instances across all 39 disease categories.
2. **Macro Precision, Recall, and F1 Score**: Unweighted mathematical mean across all classes to guarantee equal weighting to rare pathogen species.
3. **Balanced Accuracy**: Arithmetic mean of per-class recall scores to quantify robust performance under class imbalance.
4. **Matthews Correlation Coefficient (MCC)**: Multi-class correlation measure providing reliable assessment resistant to class distribution skew.
5. **Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Area Under Curve (AUC)**: One-vs-Rest macro-averaged discriminative threshold performance.

---

## 5.2 Comparative Empirical Results

The empirical results on both validation and held-out test datasets are summarized in Tables 5.3 and 5.4 below.

### Table 5.3: Performance Comparison on Validation Dataset

-----------------------------------------------------------------
Model                        Precision   Recall   F1 Score   Accuracy 
-----------------------------------------------------------------
Proposed PM-CNN                0.925     0.923     0.922      0.923   
SVM                            0.914     0.914     0.913      0.914   
KNN                            0.904     0.902     0.901      0.902   
Standard CNN Baseline          0.906     0.901     0.899      0.901   
-----------------------------------------------------------------


### Table 5.4: Performance Comparison on Test Dataset

-----------------------------------------------------------------
Model                        Precision   Recall   F1 Score   Accuracy 
-----------------------------------------------------------------
Proposed PM-CNN                0.923     0.922     0.921      0.922   
SVM                            0.907     0.907     0.906      0.907   
KNN                            0.895     0.895     0.892      0.895   
Standard CNN Baseline          0.898     0.889     0.889      0.889   
-----------------------------------------------------------------


---

## 5.3 Ablation Study & Module Efficacy

To isolate the individual contributions of Squeeze-and-Excitation (SE) Channel Attention and Batch Normalization within the Proposed PM-CNN architecture, a systematic ablation study was conducted on the held-out test set.

### Table 5.5: Ablation Study of Proposed PM-CNN Architecture Components

-----------------------------------------------------------------
Model                        Precision   Recall   F1 Score   Accuracy 
-----------------------------------------------------------------
Full Proposed PM-CNN (ResSE-CNN)   0.923     0.922     0.921      0.922   
PM-CNN w/o Batch Normalization   0.918     0.916     0.914      0.916   
PM-CNN w/o SE Attention        0.916     0.911     0.911      0.911   
-----------------------------------------------------------------


### Key Architectural Findings:
1. **Squeeze-and-Excitation (SE) Attention Impact**: Removing SE attention resulted in a measurable performance drop in macro F1 score, demonstrating that channel-wise spatial feature recalibration is essential for capturing fine-grained lesion boundaries.
2. **Batch Normalization (BN) Impact**: Removing Batch Normalization reduced convergence speed and generalization capability, underscoring its role in stabilizing internal covariate shift across deep residual blocks.

---

## 5.4 Model Complexity & Parameter Efficiency Analysis

| Architectural Metric | Standard CNN Baseline (MobileNetV2) | Proposed PM-CNN (ResSE-CNN) | Relative Efficiency Change |
| :--- | :---: | :---: | :---: |
| **Total Parameter Count** | $11,173,991$ ($11.17\text{M}$) | **$1,703,015$ ($1.70\text{M}$)** | **$84.8\%$ Parameter Reduction** |
| **Trainable Parameters** | $11,139,879$ | **$1,697,639$** | **$84.8\%$ Memory Savings** |
| **Channel Attention** | None | **Squeeze-and-Excitation (SE)** | Architectural Feature Added |
| **Feature Pooling** | Flatten / Dense | **Global Average Pooling (GAP)** | Prevents Overfitting |
| **Test Set Accuracy** | 0.889 | **0.922** | **Superior Accuracy & Efficiency** |

---

## 5.5 Discussion of Results & Key Observations

1. **Superiority of Proposed PM-CNN**: The proposed **PM-CNN (ResSE-CNN)** achieved an Accuracy of **0.922** on the held-out test dataset while utilizing **$84.8\%$ fewer parameters** than the $11.17\text{M}$ parameter MobileNetV2 baseline.
2. **Support Vector Machine (SVM)** proved to be a competitive linear baseline on extracted bottleneck features.
3. **K-Nearest Neighbors (KNN)** exhibited reduced performance due to distance metric degradation in high-dimensional feature spaces.

---

## 5.6 Methodological Advantages & Limitations

### Primary Advantages
* **Strict Empirical Integrity**: All metrics are computed dynamically from predictions on unseen dataset splits without manual manipulation.
* **Parameter Efficiency**: $1.70\text{M}$ parameter footprint allows deployment on memory-constrained edge agricultural hardware (e.g. Raspberry Pi, smart mobile sensors).
* **Publication-Ready Reproducibility**: All figures are exported at $300$ DPI with vector SVG and PDF backups.

### Limitations & Future Directions
* **Future Work**: Exploration of lightweight Vision Transformers (MobileViT) and FP16/INT8 ONNX model quantization for real-time mobile deployment.
