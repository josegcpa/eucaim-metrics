import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from .base import AbstractMetrics


class ClassificationMetrics(AbstractMetrics):
    """
    Class to compute classification metrics.
    """

    @property
    def metric_match(self):
        return {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "auroc": roc_auc_score,
            "average_precision": average_precision_score,
        }

    def calculate_metrics(
        self,
        pred: list[float] | np.ndarray,
        gt: list[float | int] | np.ndarray,
        metrics: list[str] | None = None,
        ci: float = 0.95,
    ) -> dict:
        if metrics is None:
            metrics = self.metric_match.keys()
        output_dict = {}
        for metric in metrics:
            if metric in self.metric_match:
                output_dict[metric] = self.metric_match[metric](
                    gt, pred, **self.params[metric]
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return output_dict
