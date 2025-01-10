import numpy as np
from eucaim_eval.classification import ClassificationMetrics

rng = np.random.default_rng(42)
N = 1000


def test_classification_binary_metrics():
    classification_metrics = ClassificationMetrics(n_workers=4)
    pred = rng.random(N)
    gt = rng.choice([0, 1], N)

    metrics = classification_metrics.calculate_metrics(pred, gt)


def test_classification_multiclass_metrics():
    classification_metrics = ClassificationMetrics(
        n_classes=3, input_is_one_hot=False, n_workers=4
    )
    pred = rng.random((N, 3))
    pred /= pred.sum(axis=1, keepdims=True)
    gt = rng.choice([0, 1, 2], N)

    metrics = classification_metrics.calculate_metrics(pred, gt)


def test_classification_multiclass_metrics_alt():
    classification_metrics = ClassificationMetrics(
        n_classes=3, input_is_one_hot=True, n_workers=4
    )
    pred = rng.random((N, 3))
    pred /= pred.sum(axis=1, keepdims=True)
    gt = np.eye(3)[rng.choice([0, 1, 2], N)]

    metrics = classification_metrics.calculate_metrics(pred, gt)
