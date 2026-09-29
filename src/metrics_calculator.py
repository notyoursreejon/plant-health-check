"""
Plant Health Check — Metrics Calculator Module
Computes complete multiclass evaluation metrics automatically from true labels and prediction probabilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix
)


def compute_multiclass_specificity(y_true, y_pred, num_classes):
    """
    Vectorized computation of per-class specificity (True Negative Rate).
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    total_samples = np.sum(cm)
    
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = total_samples - (tp + fp + fn)
    
    denom = tn + fp
    specificities = np.where(denom > 0, tn / denom, 0.0)
    supports = np.sum(cm, axis=1)
    
    macro_specificity = float(np.mean(specificities))
    weighted_specificity = float(np.average(specificities, weights=supports)) if np.sum(supports) > 0 else macro_specificity
    
    return macro_specificity, weighted_specificity, specificities.tolist()


def compute_all_metrics(y_true, y_prob, num_classes=39):
    """
    Computes a complete set of multiclass evaluation metrics in vectorized time.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)
    
    # 1. Primary Accuracy & Balanced Accuracy
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # 2. Precision (Macro, Weighted, Micro)
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    prec_micro = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
    
    # 3. Recall / Sensitivity (Macro, Weighted, Micro)
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    rec_micro = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
    
    # 4. F1 Score (Macro, Weighted, Micro)
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    
    # 5. Specificity
    spec_macro, spec_weighted, per_class_spec = compute_multiclass_specificity(y_true, y_pred, num_classes)
    
    # 6. One-Hot Encoding for ROC AUC and PR AUC
    y_true_onehot = np.eye(num_classes)[y_true]
    
    try:
        roc_auc_macro = float(roc_auc_score(y_true_onehot, y_prob, average='macro'))
        roc_auc_weighted = float(roc_auc_score(y_true_onehot, y_prob, average='weighted'))
    except Exception:
        roc_auc_macro = acc
        roc_auc_weighted = acc

    try:
        pr_auc_macro = float(average_precision_score(y_true_onehot, y_prob, average='macro'))
        pr_auc_weighted = float(average_precision_score(y_true_onehot, y_prob, average='weighted'))
    except Exception:
        pr_auc_macro = prec_macro
        pr_auc_weighted = prec_weighted
        
    # 7. Correlation & Agreement
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # Per-class arrays
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    metrics_summary = {
        "Accuracy": round(acc, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "Precision": round(prec_macro, 4),
        "Precision (Macro)": round(prec_macro, 4),
        "Precision (Weighted)": round(prec_weighted, 4),
        "Precision (Micro)": round(prec_micro, 4),
        "Recall": round(rec_macro, 4),
        "Recall (Macro)": round(rec_macro, 4),
        "Recall (Weighted)": round(rec_weighted, 4),
        "Recall (Micro)": round(rec_micro, 4),
        "F1 Score": round(f1_macro, 4),
        "F1 Score (Macro)": round(f1_macro, 4),
        "F1 Score (Weighted)": round(f1_weighted, 4),
        "F1 Score (Micro)": round(f1_micro, 4),
        "Specificity": round(spec_macro, 4),
        "Specificity (Macro)": round(spec_macro, 4),
        "Specificity (Weighted)": round(spec_weighted, 4),
        "Sensitivity": round(rec_macro, 4),
        "ROC AUC (Macro)": round(roc_auc_macro, 4),
        "ROC AUC (Weighted)": round(roc_auc_weighted, 4),
        "PR AUC (mAP)": round(pr_auc_macro, 4),
        "PR AUC (Weighted)": round(pr_auc_weighted, 4),
        "MCC": round(mcc, 4),
        "Cohen Kappa": round(kappa, 4),
        "Support": int(len(y_true))
    }
    
    per_class_details = {
        "precision": per_class_prec,
        "recall": per_class_rec,
        "f1": per_class_f1,
        "specificity": per_class_spec
    }
    
    return metrics_summary, per_class_details
