"""
Implements a set of metrics for segmentation in medical imaging.

Based on the [1] and [2].

[1] https://www.nature.com/articles/s41592-023-02151-z
[2] https://github.com/Project-MONAI/MetricsReloaded
"""

import numpy as np
from dataclasses import dataclass
from functools import cache
from typing import Callable, Iterator
from scipy import optimize, ndimage
from tqdm import tqdm
from skimage.morphology import binary_erosion
from scipy.spatial import distance
from .base import ImabeBasedMetrics, ImageMultiFormat


@dataclass
class SegmentationMetrics(ImabeBasedMetrics):
    """
    Class to compute segmentation metrics.

    Args:
        match_regions (bool, optional): whether to match predicted and target
            regions. Defaults to False.
    """

    match_regions: bool = False

    @property
    def metric_match(self):
        return {
            "dice": self.dice_score,
            "iou": self.iou,
            "hausdorff_distance": self.hausdorff_distance,
            "normalised_surface_distance": self.normalised_surface_distance,
        }

    @cache(max_size=None)
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

    @cache(max_size=None)
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

    @cache(max_size=None)
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

    @cache(max_size=None)
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

    @cache(max_size=None)
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

    @cache(max_size=128)
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

    @cache(max_size=None)
    def dice_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute the Dice score between the predicted and target images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.

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

    @cache(max_size=None)
    def iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute the intersection over union between the predicted and target
        images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.

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

    def binary_hausdorff_distance(
        self,
        pred_surface: np.ndarray,
        gt_surface: np.ndarray,
        q: float = 1.0,
    ) -> float:
        """
        Compute the Hausdorff distance between the predicted and target
        binary images.

        Args:
            pred (np.ndarray): predicted image.
            gt (np.ndarray): target image.
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

    @cache(max_size=None)
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

        pred_surface = self.__surface(pred)
        gt_surface = self.__surface(gt)

        output = self.calculate_across_classes_if_necessary(
            pred_surface, gt_surface, self.binary_hausdorff_distance, q=q
        )

        return self.reduce_if_necessary(output)

    def binary_normalised_surface_distance(
        self,
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

    @cache(max_size=None)
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

        if max_distance is None:
            raise ValueError(
                "max_distance must be provided for normalised surface distance."
            )
        pred_surface = self.__surface(pred)
        gt_surface = self.__surface(gt)

        output = self.calculate_across_classes_if_necessary(
            pred_surface,
            gt_surface,
            self.binary_normalised_surface_distance,
            max_distance=max_distance,
        )
        return self.reduce_if_necessary(output)

    def __match_and_iterate_regions(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Assign predicted regions to their respective ground truth regions based
        on overlap. Essentially, a predicted region is assigned to a ground
        truth if the overlap between both is the maximum within the predicted
        region's overlap values. This is performed first by determining the
        available connected component regions, and then by performing a simple
        linear sum assignment.

        This may lead to predicted regions with no ground truth region, or
        ground truth regions with no predicted region. These are considered
        as false positives and false negatives, respectively, and are returned
        in any case with an empty image.

        Args:
            pred (np.ndarray): predicted image.
            gt (np.ndarray): target image.

        Returns:
            Iterator[tuple[np.ndarray, np.ndarray]]: iterator of correctly
                assigned tuples of predicted and ground truth regions.
        """
        pred_labels, n_pred = ndimage.label(pred)
        gt_labels, n_gt = ndimage.label(gt)

        cost_matrix = np.zeros((n_pred, n_gt))
        for i in range(1, n_pred + 1):
            pred_mask = pred_labels == i
            for j in range(1, n_gt + 1):
                gt_mask = gt_labels == j
                overlap = np.sum(pred_mask & gt_mask)
                cost_matrix[i - 1, j - 1] = -overlap

        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

        for i in range(n_pred):
            pred_mask = pred_labels == (i + 1)
            if i in row_ind:
                j = col_ind[np.where(row_ind == i)[0][0]]
                gt_mask = gt_labels == (j + 1)
            else:
                gt_mask = np.zeros_like(gt)
            yield pred_mask, gt_mask

        for j in range(n_gt):
            if j not in col_ind:
                gt_mask = gt_labels == (j + 1)
                pred_mask = np.zeros_like(pred)
                yield pred_mask, gt_mask

    def __center(self, array: np.ndarray) -> np.ndarray:
        """
        Computes the center of an object in an array.

        Args:
            array (np.ndarray): array.

        Returns:
            np.ndarray: center of the positive pixels in the array.
        """
        return ndimage.center_of_mass(array)

    def calculate_case(
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
        output_dict = {k: [] for k in metrics}
        output_dict.update(
            {"idx_pred": [], "idx_gt": [], "center_pred": [], "center_gt": []}
        )
        pred_gt_iterator = (
            self.__match_and_iterate_regions(pred, gt)
            if self.match_regions
            else [(pred, gt, 0, 0, None, None)]
        )
        for matched_pred, matched_gt, idx_pred, idx_gt in pred_gt_iterator:
            output_dict["idx_pred"].append(idx_pred)
            output_dict["idx_gt"].append(idx_gt)
            for metric in metrics:
                if metric in self.metric_match:
                    output_dict[metric] = self.metric_match[metric](
                        matched_pred, matched_gt, **self.params[metric]
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            output_dict["center_pred"].append(self.__center(matched_pred))
            output_dict["center_gt"].append(self.__center(matched_gt))
        return output_dict

    def calculate_metrics_standard(
        self,
        preds: list[ImageMultiFormat],
        gts: list[ImageMultiFormat],
        metrics: list[str],
        ci: float = 0.95,
    ) -> list[dict]:
        """
        Calculate metrics for multiple predictions and ground truth pairs and
        returns a structured dictionary with the output.

        Args:
            preds (list[ImageMultiFormat]): predictions.
            gts (list[ImageMultiFormat]): ground truths.

        Returns:
            list[dict]: metrics.

        Raises:
            ValueError: if preds and gts have different lengths.
        """
        if len(preds) != len(gts):
            raise ValueError("preds and gts must have the same length")
        output = []
        average_values = {metric: [] for metric in metrics}
        n = 0
        q = (1 - ci) / 2
        q = q, 1 - q
        for i, (pred, gt) in tqdm(enumerate(zip(preds, gts)), total=len(preds)):
            metrics = self.calculate_case(pred, gt, metrics)
            metrics["pred_path"] = pred if isinstance(pred, str) else str(i)
            metrics["gt_path"] = gt if isinstance(gt, str) else str(i)
            for metric in metrics:
                average_values[metric].append(metrics[metric])
            n += 1
            output.append(metrics)
        output_dict = {"metrics": output}
        output_dict["metrics_mean"] = {
            metric: np.mean(average_values[metric], axis=-1)
            for metric in average_values
        }
        output_dict["metrics_median"] = {
            metric: np.median(average_values[metric], axis=-1)
            for metric in average_values
        }
        output_dict["metrics_sd"] = {
            metric: np.std(average_values[metric], axis=-1)
            for metric in average_values
        }
        output_dict["metrics_ci"] = {
            **{
                metric: np.quantile(average_values[metric], q=q, axis=-1)
                for metric in average_values
            },
            "ci": ci,
        }
        return output_dict

    def calculate_metrics_with_match_regions(
        self,
        preds: list[ImageMultiFormat],
        gts: list[ImageMultiFormat],
        metrics: list[str],
        ci: float = 0.95,
    ) -> list[dict]:
        """
        Calculate metrics for multiple predictions and ground truth pairs and
        returns a structured dictionary with the output.

        Args:
            preds (list[ImageMultiFormat]): predictions.
            gts (list[ImageMultiFormat]): ground truths.

        Returns:
            list[dict]: metrics.

        Raises:
            ValueError: if preds and gts have different lengths.
        """
        if len(preds) != len(gts):
            raise ValueError("preds and gts must have the same length")
        output = []
        average_values = {metric: [] for metric in metrics}
        n = 0
        q = (1 - ci) / 2
        q = q, 1 - q
        real_n_classes = self.n_classes if self.n_classes > 2 else 1
        self.n_classes = 2
        for i, (pred, gt) in tqdm(enumerate(zip(preds, gts)), total=len(preds)):
            for cl in range(real_n_classes):
                metrics = self.calculate_case(pred[cl], gt[cl], metrics)
                metrics["pred_path"] = pred if isinstance(pred, str) else str(i)
                metrics["gt_path"] = gt if isinstance(gt, str) else str(i)
                for metric in metrics:
                    average_values[metric].extend(metrics[metric])
                n += 1
                output.append(metrics)
        output_dict = {"metrics": output}
        output_dict["metrics_mean"] = {
            metric: np.mean(average_values[metric], axis=-1)
            for metric in average_values
        }
        output_dict["metrics_median"] = {
            metric: np.median(average_values[metric], axis=-1)
            for metric in average_values
        }
        output_dict["metrics_sd"] = {
            metric: np.std(average_values[metric], axis=-1)
            for metric in average_values
        }
        output_dict["metrics_ci"] = {
            **{
                metric: np.quantile(average_values[metric], q=q, axis=-1)
                for metric in average_values
            },
            "ci": ci,
        }

        self.n_classes = real_n_classes
        return output_dict

    def calculate_metrics(
        self,
        preds: list[ImageMultiFormat],
        gts: list[ImageMultiFormat],
        metrics: list[str],
        ci: float = 0.95,
        match_regions: bool = False,
    ) -> list[dict]:
        """
        Calculate metrics for multiple predictions and ground truth pairs and
        returns a structured dictionary with the output. Region matching (when
        predicted and ground truth connected components are matched and metrics
        are calculated only between matching regions) is supported.

        Args:
            preds (list[ImageMultiFormat]): predictions.
            gts (list[ImageMultiFormat]): ground truths.
            metrics (list[str]): metrics to calculate.
            ci (float, optional): confidence interval. Defaults to 0.95.
            match_regions (bool, optional): if True, predicted and ground truth
                regions will be matched. Defaults to False.

        Returns:
            list[dict]: metrics.
        """
        if match_regions:
            return self.calculate_metrics_with_match_regions(
                preds, gts, metrics, ci
            )
        else:
            return self.calculate_metrics_standard(preds, gts, metrics, ci)
