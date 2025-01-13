from multiprocessing import Pool

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base import AbstractMetrics


class ClassificationMetrics(AbstractMetrics):
    """
    Class to compute classification metrics.
    """

    @property
    def metric_match(self):
        return {
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "auroc": self._auroc,
            "ap": self._ap,
        }

    def _precision(
        self, gt: np.ndarray, pred: np.ndarray
    ) -> float | np.ndarray:
        if self.n_classes > 2:
            return precision_score(gt, pred, average=None)
        return precision_score(gt, pred)

    def _recall(self, gt: np.ndarray, pred: np.ndarray) -> float | np.ndarray:
        if self.n_classes > 2:
            return recall_score(gt, pred, average=None)
        return recall_score(gt, pred)

    def _f1(self, gt: np.ndarray, pred: np.ndarray) -> float | np.ndarray:
        if self.n_classes > 2:
            return f1_score(gt, pred, average=None)
        return f1_score(gt, pred)

    def _auroc(self, gt: np.ndarray, pred: np.ndarray) -> float | np.ndarray:
        if self.n_classes > 2:
            return roc_auc_score(gt, pred, multi_class="ovr", average=None)
        return roc_auc_score(gt, pred)

    def _ap(self, gt: np.ndarray, pred: np.ndarray) -> float | np.ndarray:
        if self.n_classes > 2:
            if len(gt.shape) == 1:
                gt = self.to_one_hot(gt)
            return average_precision_score(gt, pred, average=None)
        return average_precision_score(gt, pred)

    @property
    def proba_metrics(self):
        return ["auroc", "ap"]

    def calculate_metrics(
        self,
        preds: list[float] | np.ndarray,
        gts: list[float | int] | np.ndarray,
        metrics: list[str] | None = None,
        ci: float = 0.95,
    ) -> dict:
        """
        Calculate classification metrics.

        Args:
            preds (list[float] | np.ndarray): prediction probabilities.
            gts (list[float  |  int] | np.ndarray): target variables.
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
        output_dict = {}
        if self.n_classes == 2:
            preds, proba = preds > 0.5, preds
        else:
            proba = preds
            preds = np.argmax(preds, axis=1)
            if self.input_is_one_hot is True:
                gts = np.argmax(gts, axis=1)
        pool = Pool(self.n_workers) if self.n_workers > 1 else None
        for metric in metrics:
            if metric in self.metric_match:
                p = proba if metric in self.proba_metrics else preds
                output_dict[metric] = {}
                output_dict[metric]["value"] = self.metric_match[metric](
                    gts, p, **self.params[metric]
                )
                bootstrap_sample = self.bootstrap(
                    arrays=[gts, p],
                    fn=self.metric_match[metric],
                    n_bootstraps=1000,
                    bootstrap_size=0.5,
                    mp_pool=pool,
                )
                output_dict[metric]["mean"] = np.mean(bootstrap_sample)
                output_dict[metric]["median"] = np.median(bootstrap_sample)
                output_dict[metric]["sd"] = np.std(bootstrap_sample)
                output_dict[metric]["ci"] = np.quantile(bootstrap_sample, q)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        pool.close()
        pool.join()
        return output_dict
