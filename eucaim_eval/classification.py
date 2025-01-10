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

    @property
    def proba_metrics(self):
        return ["auroc", "average_precision"]

    def calculate_metrics(
        self,
        pred: list[float] | np.ndarray,
        gt: list[float | int] | np.ndarray,
        metrics: list[str] | None = None,
        ci: float = 0.95,
    ) -> dict:
        """
        Calculate classification metrics.

        Args:
            pred (list[float] | np.ndarray): prediction probabilities.
            gt (list[float  |  int] | np.ndarray): target variables.
            metrics (list[str] | None, optional): list of metric strings.
                Defaults to None.
            ci (float, optional): confidence intervals. Defaults to 0.95.

        Raises:
            ValueError: if pred and gt have different lengths.

        Returns:
            dict: dictionary with entries for "metrics", which are calculated
                from the raw data, and "metrics_mean", "metrics_median",
                "metrics_sd", and "metrics_ci", which are calculated using
                bootstraping.
        """
        if metrics is None:
            metrics = self.metric_match.keys()
        q = (1 - ci) / 2
        q = q, 1 - q
        output_dict = {
            "metrics": {},
            "metrics_mean": {},
            "metrics_median": {},
            "metrics_sd": {},
            "metrics_ci": {},
        }
        if self.n_classes == 2:
            pred, proba = pred > 0.5, pred
        else:
            proba = pred
            pred = np.argmax(pred, axis=1)
            if len(gt.shape) > pred.shape:
                gt = np.argmax(gt, axis=1)
        for metric in metrics:
            if metric in self.metric_match:
                p = proba if metric in self.proba_metrics else pred
                output_dict["metrics"][metric] = self.metric_match[metric](
                    gt, p, **self.params[metric]
                )
                bootstrap_sample = self.bootstrap(
                    self.metric_match[metric], 1000, 0.5, gt, p
                )
                output_dict["metrics_mean"][metric] = np.mean(bootstrap_sample)
                output_dict["metrics_median"][metric] = np.median(
                    bootstrap_sample
                )
                output_dict["metrics_sd"][metric] = np.std(bootstrap_sample)
                output_dict["metrics_ci"][metric] = np.quantile(
                    bootstrap_sample, q
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return output_dict
