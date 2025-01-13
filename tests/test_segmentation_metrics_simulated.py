import numpy as np
from pytest import approx
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


binary_output = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
binary_output_pred = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

mc_output = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 3, 3, 3, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
)
mc_output_pred = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 3, 3, 3, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]
)

real_values = {
    "multiclass": {
        "match_region": {
            "dice": {
                0: 0.9838709677419355,
                1: 0.8660714285714286,
                2: 0.8,
                3: 1.0,
            },
            "hd": {
                0: 1.0,
                1: 1.0,
                2: 1.0,
                3: 0.0,
            },
            "normalised_surface_distance": {
                0: 0.9682539682539683,
                1: 0.8571428571428571,
                2: 0.8,
                3: 1.0,
            },
        },
        "standard": {
            "dice": {
                0: 0.9838709677419355,
                1: 0.8695652173913043,
                2: 0.8,
                3: 1.0,
            },
            "hd": {
                0: 1.0,
                1: 1.0,
                2: 1.0,
                3: 0.0,
            },
            "normalised_surface_distance": {
                0: 0.9682539682539683,
                1: 0.8571428571428571,
                2: 0.8,
                3: 1.0,
            },
        },
    }
}


def test_segmentation_binary_metrics_standard():
    output, output_pred = binary_output, binary_output_pred
    classification_metrics = SegmentationMetrics(
        n_workers=4,
        params={"normalised_surface_distance": {"max_distance": 0.5}},
        n_classes=output.max() + 1,
        input_is_one_hot=False,
    )

    metrics = classification_metrics.calculate_metrics([output_pred], [output])


def test_segmentation_mc_metrics_standard():
    output, output_pred = mc_output, mc_output_pred
    classification_metrics = SegmentationMetrics(
        n_workers=4,
        params={"normalised_surface_distance": {"max_distance": 0.5}},
        n_classes=output.max() + 1,
        input_is_one_hot=False,
    )

    metrics = classification_metrics.calculate_metrics([output_pred], [output])

    gt = real_values["multiclass"]["standard"]
    for k in gt["dice"].keys():
        assert metrics["dice"]["mean"][k] == approx(gt["dice"][k])
        assert metrics["hausdorff_distance"]["mean"][k] == approx(gt["hd"][k])
        assert metrics["normalised_surface_distance"]["mean"][k] == approx(
            gt["normalised_surface_distance"][k]
        )


def test_segmentation_binary_metrics_match_region():
    output, output_pred = binary_output, binary_output_pred
    classification_metrics = SegmentationMetrics(
        n_workers=4,
        params={"normalised_surface_distance": {"max_distance": 0.5}},
        n_classes=output.max() + 1,
        input_is_one_hot=False,
    )

    metrics = classification_metrics.calculate_metrics(
        [output_pred], [output], match_regions=True
    )


def test_segmentation_mc_metrics_match_region():
    output, output_pred = mc_output, mc_output_pred
    classification_metrics = SegmentationMetrics(
        n_workers=4,
        params={"normalised_surface_distance": {"max_distance": 0.5}},
        n_classes=output.max() + 1,
        input_is_one_hot=False,
    )

    metrics = classification_metrics.calculate_metrics(
        [output_pred], [output], match_regions=True
    )

    gt = real_values["multiclass"]["match_region"]
    for k in gt["dice"].keys():
        assert metrics["dice"]["mean"][k] == approx(gt["dice"][k])
        assert metrics["hausdorff_distance"]["mean"][k] == approx(gt["hd"][k])
        assert metrics["normalised_surface_distance"]["mean"][k] == approx(
            gt["normalised_surface_distance"][k]
        )
