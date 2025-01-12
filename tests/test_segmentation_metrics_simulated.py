import numpy as np
from eucaim_eval.segmentation import SegmentationMetrics
from skimage.draw import random_shapes
from scipy.ndimage import binary_erosion

rng = np.random.default_rng(42)
N = 1000
SH = (128, 128)
N_SLICES = 5


def generate_data(n_classes: int = 2):
    intensity_range = [i for i in range(1, n_classes + 1)]
    shapes = random_shapes(
        SH,
        max_shapes=10,
        intensity_range=intensity_range,
        allow_overlap=False,
        min_shapes=2,
        num_channels=1,
    )[0][:, :, 0]
    shapes = np.where(shapes > n_classes, 0, shapes)
    eroded_shapes = sum(
        [
            binary_erosion(np.where(shapes == idx, 1, 0), iterations=1) * idx
            for idx in range(1, n_classes)
        ]
    )
    output = np.zeros(SH + (N_SLICES,), dtype=np.uint8)
    output[:, :, 3] = shapes

    output_pred = np.zeros(SH + (N_SLICES,), dtype=np.uint8)
    output_pred[:, :, 3] = eroded_shapes

    return output, output_pred


output, output_pred = generate_data(n_classes=2)


def test_segmentation_binary_metrics():
    classification_metrics = SegmentationMetrics(
        n_workers=4,
        params={"normalised_surface_distance": {"max_distance": 2}},
        n_classes=2,
    )

    metrics = classification_metrics.calculate_metrics([output_pred], [output])

    print(metrics)
