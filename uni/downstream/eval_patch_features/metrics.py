from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    precision_score,
    recall_score,
    confusion_matrix
)

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.

    Args:
        targets_all (array-like): True target values.
        preds_all (array-like): Predicted target values.
        probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
        get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
        prefix (str, optional): Prefix to add to the result keys. Defaults to "".
        roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.

    Returns:
        dict: Dictionary containing the evaluation metrics.

    """
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

    precision = precision_score(targets_all, preds_all, average='weighted', zero_division=0)
    recall = recall_score(targets_all, preds_all, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(targets_all, preds_all)
    
    # Sensitivity (Recall for positive class)
    sensitivity = recall_score(targets_all, preds_all, pos_label=1, zero_division=0)
    #sensitivity = recall_score(targets_all, preds_all, pos_label=1, average='micro', zero_division=0)

    
    # Specificity (Recall for negative class)
    specificity = recall_score(targets_all, preds_all, pos_label=0, zero_division=0)

    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}sensitivity": sensitivity,
        f"{prefix}specificity": specificity,
        f"{prefix}confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON compatibility
    }

    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep

    if probs_all is not None:
        roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
        eval_metrics[f"{prefix}auroc"] = roc_auc

    return eval_metrics

def print_metrics(eval_metrics):
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        
        if isinstance(v, float) or isinstance(v, int):
            print(f"Test {k}: {v:.3f}")
        else:
            print(f"Test {k}: {v}")

# Example usage:
# targets_all = [true_labels]
# preds_all = [predicted_labels]
# probs_all = [predicted_probabilities] # If applicable

# metrics = get_eval_metrics(targets_all, preds_all, probs_all, prefix="test_", get_report=True)
# print_metrics(metrics)
