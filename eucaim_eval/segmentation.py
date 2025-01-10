"""
Implements a set of metrics for segmentation in medical imaging.

Based on the [1] and [2].

[1] https://www.nature.com/articles/s41592-023-02151-z
[2] https://github.com/Project-MONAI/MetricsReloaded
"""

import numpy as np
import SimpleITK as sitk
from dataclasses import dataclass
from skimage.morphology import binary_erosion
from scipy.spatial import distance
from functools import lru_cache
from typing import Callable

ImageMultiFormat = str | sitk.Image


@dataclass
class ClassificationMetrics:
    """
    Class to compute classification metrics.
    """

    n_classes: int = 2
    reduction: None | str | Callable = None
    is_input_one_hot: bool = False

    def load_image(self, image: ImageMultiFormat) -> sitk.Image:
        """
        Load the image.

        Args:
            image (ImageMultiFormat): Image to load.

        Returns:
            sitk.Image: Loaded image.
        """
        if isinstance(image, str):
            image = sitk.ReadImage(image)
        return image

    def check_images(self, *images: list[sitk.Image]) -> None:
        """
        Check if the images have the same size and spacing.

        Args:
            image_1 (sitk.Image): First image.
            image_2 (sitk.Image): Second image.

        Returns:
                bool: True if the images have the same size and spacing, False otherwise.

        Raises:
            ValueError: If the images have different sizes or spacings.
        """
        if len(image) > 1:
            for image in images[1:]:
                if image.GetSize() != image[0].GetSize():
                    raise ValueError("Images must have the same size.")
                if image.GetSpacing() != image[0].GetSpacing():
                    raise ValueError("Images must have the same spacing.")

    def load_images(self, *images: list[ImageMultiFormat]) -> None:
        """
        Load the images.

        Args:
            *images (ImageMultiFormat): Images to load.
        """
        images = [self.load_image(image) for image in images]
        self.check_images(*images)
        return images

    def to_one_hot(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the image to one-hot encoding.

        Args:
            image (np.ndarray): Image to convert.

        Returns:
            np.ndarray: One-hot encoded image.
        """
        return np.eye(self.n_classes)[image.astype(int)]

    def load_arrays(self, *images: list[ImageMultiFormat]) -> list[np.ndarray]:
        """
        Load the images as numpy arrays.

        Args:
            *images (ImageMultiFormat): Images to load.

        Returns:
            list[np.ndarray]: Loaded images.
        """
        if isinstance(images[0], np.ndarray):
            output = images
        else:
            output = [
                sitk.GetArrayFromImage(image)
                for image in self.load_images(*images)
            ]
        if (self.is_input_one_hot is False) and (self.n_classes > 2):
            output = [self.to_one_hot(image) for image in output]
        return output

    def reduce_if_necessary(
        self, metric_array: np.ndarray
    ) -> float | np.ndarray:
        """
        Reduce the metric array if necessary.

        Args:
            metric_array (np.ndarray): Metric array.

        Returns:
            float | np.ndarray: Reduced metric array.
        """
        if self.reduction is None:
            return metric_array
        elif isinstance(self.reduction, str):
            if self.reduction == "mean":
                return np.mean(metric_array)
            elif self.reduction == "sum":
                return np.sum(metric_array)
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
        elif callable(self.reduction):
            return self.reduction(metric_array)


@dataclass
class SegmentationMetrics(ClassificationMetrics):
    """
    Class to compute binary metrics.
    """

    params: dict = None

    def __post_init__(self):
        self.params = self.params or {}
        for metric in self.metric_match:
            if metric not in self.params:
                self.params[metric] = {}

    @property
    def metric_match(self):
        return {
            "dice": self.dice_score,
            "iou": self.iou,
            "hausdorff_distance": self.hausdorff_distance,
            "normalised_surface_distance": self.normalised_surface_distance,
        }

    @lru_cache(max_size=None)
    def __intersection_binary(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> float:
        """
        Compute the intersection between two images each with a single class.

        Args:
            image_1 (np.ndarray): First image.
            image_2 (np.ndarray): Second image.

        Returns:
            float: Intersection between the two images.
        """
        return np.sum(image_1 * image_2)

    @lru_cache(max_size=None)
    def __union_binary(self, image_1: np.ndarray, image_2: np.ndarray) -> float:
        """
        Compute the union between two images each with a single class.

        Args:
            image_1 (np.ndarray): First image.
            image_2 (np.ndarray): Second image.

        Returns:
            float: Union between the two images.
        """
        return np.sum((image_1 + image_2) > 0)

    @lru_cache(max_size=None)
    def __intersection_multiclass(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> float:
        """
        Compute the intersection between two images with multiple classes. In
        other words, assumes the images are one-hot encoded with the first
        channel corresponding to the classes.

        Args:
            image_1 (np.ndarray): First image.
            image_2 (np.ndarray): Second image.

        Returns:
            float: Intersection between the two images.
        """
        return np.sum(
            image_1.reshape(self.n_classes, -1)
            == image_2.reshape(self.n_classes, -1),
            axis=-1,
        )

    @lru_cache(max_size=None)
    def __union_multiclass(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> float:
        """
        Compute the union between two images with multiple classes. In other
        words, assumes the images are one-hot encoded with the first channel
        corresponding to the classes.

        Args:
            image_1 (np.ndarray): First image.
            image_2 (np.ndarray): Second image.

        Returns:
            float: Intersection between the two images.
        """
        return np.sum(
            (
                image_1.reshape(self.n_classes, -1)
                + image_2.reshape(self.n_classes, -1)
            )
            > 0,
            axis=-1,
        )

    @lru_cache(max_size=None)
    def __surface(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the surface of the image using simple binary erosion.

        Args:
            image (np.ndarray): Image.

        Returns:
            float: Surface of the image.
        """
        if self.n_classes == 2:
            eroded_image = binary_erosion(image)
        else:
            eroded_image = np.stack(
                [binary_erosion(image[i]) for i in range(self.n_classes)]
            )
        return image - eroded_image

    @lru_cache(max_size=128)
    def __distance(
        self, surface_1: np.ndarray, surface_2: np.ndarray
    ) -> np.ndarray:
        """
        Compute the distance between two surfaces.

        Args:
            surface_1 (np.darray): First surface.
            surface_2 (np.darray): Second surface.

        Returns:
            np.darray: Distance between the two surfaces.
        """
        coords_1 = np.where(surface_1)
        coords_2 = np.where(surface_2)
        return distance.cdist(coords_1, coords_2)

    def dice_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute the Dice score between the predicted and target images.

        Args:
            pred (ImageMultiFormat): Predicted image.
            gt (ImageMultiFormat): Target image.

        Returns:
            float: Dice score.
        """
        if self.n_classes == 2:
            intersection = self.__intersection_binary(pred, gt)
            union = self.__union_binary(pred, gt)
        else:
            intersection = self.__intersection_multiclass(pred, gt)
            union = self.__union_multiclass(pred, gt)
        output = np.where(
            union > 0, 2 * intersection / (union + intersection), 0
        )
        return self.reduce_if_necessary(output)

    def iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute the intersection over union between the predicted and target
        images.

        Args:
            pred (ImageMultiFormat): Predicted image.
            gt (ImageMultiFormat): Target image.

        Returns:
            float: Dice score.
        """
        if self.n_classes == 2:
            intersection = self.__intersection_binary(pred, gt)
            union = self.__union_binary(pred, gt)
        else:
            intersection = self.__intersection_multiclass(pred, gt)
            union = self.__union_multiclass(pred, gt)
        output = np.where(union > 0, intersection / (union), 0)
        return self.reduce_if_necessary(output)

    def calculate_across_classes_if_necessary(
        self, pred: np.ndarray, gt: np.ndarray, fn: Callable, *args, **kwargs
    ) -> float | np.ndarray:
        """
        Compute function across classes if necessary.

        Args:
            pred (ImageMultiFormat): predicted object.
            gt (ImageMultiFormat): target object.
            fn (Callable): function.
            *args: arguments to pass to the function.
            **kwargs: keyword arguments to pass to the function.

        Returns:
            float | np.ndarray: function across classes.
        """

        if self.n_classes == 2:
            output = fn(pred, gt, *args, **kwargs)
        else:
            output = np.stack(
                [
                    fn(pred[i], gt[i], *args, **kwargs)
                    for i in range(self.n_classes)
                ]
            )
        return output

    def hausdorff_distance(
        self, pred: np.ndarray, gt: np.ndarray, q: float = 1.0
    ) -> float:
        """
        Compute the Hausdorff distance between the predicted and target images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.
            q (float, optional): quantile for the Hausdorff distance. Defaults
                to 1.0 (maximum).

        Returns:
            float: Hausdorff distance.
        """

        def binary_hausdorff_distance(
            pred_surface: np.ndarray, gt_surface: np.ndarray, q: float = 1.0
        ) -> float:
            """
            Compute the Hausdorff distance between the predicted and target
            binary images.

            Args:
                pred (np.ndarray): Predicted image.
                gt (np.ndarray): Target image.
                q (float, optional): quantile for the Hausdorff distance.
                    Defaults to 1.0 (maximum).

            Returns:
                float: Hausdorff distance.
            """
            dist_mat = self.__distance(pred_surface, gt_surface)
            minimum_distance_pred = dist_mat.min(axis=1)
            minimum_distance_gt = dist_mat.min(axis=0)
            return np.maximum(
                np.quantile(minimum_distance_pred, q),
                np.quantile(minimum_distance_gt, q),
            )

        pred_surface = self.__surface(pred)
        gt_surface = self.__surface(gt)

        output = self.calculate_across_classes_if_necessary(
            pred_surface, gt_surface, binary_hausdorff_distance, q=q
        )

        return self.reduce_if_necessary(output)

    def normalised_surface_distance(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        max_distance: float = None,
    ) -> float:
        """
        Compute the normalised surface distance between the predicted and target
        images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.
            max_distance (float, optional): maximum distance. Defaults to None.

        Returns:
            float: normalised surface distance.
        """

        def binary_normalised_surface_distance(
            pred_surface: np.ndarray,
            gt_surface: np.ndarray,
            max_distance: float,
        ) -> float:
            """
            Compute the normalised surface distance between the predicted and
            target binary images.

            Args:
                pred (np.ndarray): predicted image.
                gt (np.ndarray): target image.
                max_distance (float): maximum distance.

            Returns:
                float: normalised surface distance.
            """
            dist_mat = self.__distance(pred_surface, gt_surface)
            n_pred = dist_mat.shape[0]
            minimum_distances = dist_mat.min(axis=1)
            return np.sum(minimum_distances < max_distance) / n_pred

        if max_distance is None:
            raise ValueError(
                "max_distance must be provided for normalised surface distance."
            )
        pred_surface = self.__surface(pred)
        gt_surface = self.__surface(gt)

        output = self.calculate_across_classes_if_necessary(
            pred_surface,
            gt_surface,
            binary_normalised_surface_distance,
            max_distance=max_distance,
        )
        return self.reduce_if_necessary(output)

    def calculate_metrics(
        self,
        pred: ImageMultiFormat,
        gt: ImageMultiFormat,
        metrics: list[str] = None,
    ) -> dict[str, float]:
        """
        Compute the metrics between the predicted and target images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.
            metrics (list[str]): list of metrics to compute.

        Returns:
            dict[str, float]: dictionary of metrics.
        """
        pred, gt = self.load_arrays(pred, gt)

        if metrics is None:
            metrics = self.metric_match.keys()
        output_dict = {}
        for metric in metrics:
            if metric in self.metric_match:
                output_dict[metric] = self.metric_match[metric](
                    pred, gt, **self.params[metric]
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return output_dict
