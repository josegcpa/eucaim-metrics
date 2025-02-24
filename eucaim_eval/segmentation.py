"""
Implements a set of metrics for segmentation in medical imaging.

Based on the [1] and [2].

[1] https://www.nature.com/articles/s41592-023-02151-z
[2] https://github.com/Project-MONAI/MetricsReloaded
"""

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from scipy import ndimage, optimize
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.spatial import distance
from tqdm import tqdm

from .base import ImabeBasedMetrics, ImageMultiFormat
from .caching import MethodCache
from .utils import coherce_to_non_array


@dataclass
class SegmentationMetrics(ImabeBasedMetrics):
    """
    Class to compute segmentation metrics.

    Args:
        match_regions (bool, optional): whether to match predicted and target
            regions. Defaults to False.
    """

    match_regions: bool = False
    AVAILABLE_METRICS: list[str] = field(
        default_factory=lambda: [
            "dice",
            "iou",
            "hausdorff_distance",
            "normalised_surface_distance",
        ],
        init=False,
        repr=False,
    )
    pred_preprocessing_fn: Callable = None

    @property
    def metric_match(self):
        return {
            "dice": self.dice_score,
            "iou": self.iou,
            "hausdorff_distance": self.hausdorff_distance,
            "normalised_surface_distance": self.normalised_surface_distance,
        }

    def __intersection_binary(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> np.float64:
        """
        Compute the intersection between two images each with a single class.

        Args:
            image_1 (np.ndarray): first image.
            image_2 (np.ndarray): second image.

        Returns:
            np.float64: Intersection between the two images.
        """
        return np.sum(image_1 * image_2).astype(np.float64)

    def __union_binary(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> np.float64:
        """
        Compute the union between two images each with a single class.

        Args:
            image_1 (np.ndarray): First image.
            image_2 (np.ndarray): Second image.

        Returns:
            np.float64: Union between the two images.
        """
        return np.sum(
            np.logical_or(image_1 > 0, image_2 > 0).astype(np.float64)
        )

    def __intersection_multiclass(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> np.ndarray:
        """
        Compute the intersection between two images with multiple classes. In
        other words, assumes the images are one-hot encoded with the first
        channel corresponding to the classes.

        Args:
            image_1 (np.ndarray): first image.
            image_2 (np.ndarray): second image.

        Returns:
            np.ndarray: Intersection between the two images.
        """
        return np.sum(
            np.reshape(image_1 * image_2, [self.n_classes, -1]), axis=-1
        )

    def __union_multiclass(
        self, image_1: np.ndarray, image_2: np.ndarray
    ) -> np.ndarray:
        """
        Compute the union between two images with multiple classes. In other
        words, assumes the images are one-hot encoded with the first channel
        corresponding to the classes.

        Args:
            image_1 (np.ndarray): first image.
            image_2 (np.ndarray): second image.

        Returns:
            np.ndarray: intersection between the two images.
        """
        return np.sum(
            np.reshape(image_1 + image_2, [self.n_classes, -1]) > 0, axis=-1
        )

    def __surface(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the surface of the image using simple binary erosion.

        Args:
            image (np.ndarray): Image.

        Returns:
            float: Surface of the image.
        """
        if self.n_classes == 2:
            eroded_image = binary_erosion(binary_fill_holes(image))
        else:
            eroded_image = np.stack(
                [
                    binary_erosion(binary_fill_holes(image[i]))
                    for i in range(self.n_classes)
                ]
            )
        eroded_image = eroded_image.astype(int)
        output = image - eroded_image
        output[output < 0] = 0
        return output

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
        coords_1 = np.stack(np.where(surface_1), axis=1)
        coords_2 = np.stack(np.where(surface_2), axis=1)
        return distance.cdist(coords_1, coords_2)

    def set_cache(self, maxsize: int):
        """
        Set the cache maxsize.

        Args:
            maxsize (int): maximum size of the cache.
        """

        if isinstance(self.__intersection_binary, MethodCache):
            self.__intersection_binary.set_maxsize(maxsize)
            self.__union_binary.set_maxsize(maxsize)
            self.__intersection_multiclass.set_maxsize(maxsize)
            self.__union_multiclass.set_maxsize(maxsize)
            self.__surface.set_maxsize(maxsize)
            self.__distance.set_maxsize(maxsize)
        else:
            self.__intersection_binary = MethodCache(
                self.__intersection_binary, maxsize=maxsize
            )
            self.__union_binary = MethodCache(
                self.__union_binary, maxsize=maxsize
            )
            self.__intersection_multiclass = MethodCache(
                self.__intersection_multiclass, maxsize=maxsize
            )
            self.__union_multiclass = MethodCache(
                self.__union_multiclass, maxsize=maxsize
            )
            self.__surface = MethodCache(self.__surface, maxsize=maxsize)
            self.__distance = MethodCache(self.__distance, maxsize=maxsize)

    def dice_score(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> float | np.ndarray:
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

        output = 2 * intersection / (union + intersection)
        return self.reduce_if_necessary(output)

    def iou(self, pred: np.ndarray, gt: np.ndarray) -> float | np.ndarray:
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
        output = np.where(union > 0, intersection / union, 0)
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
        if dist_mat.size == 0:
            return np.nan
        minimum_distance_pred = dist_mat.min(axis=1)
        minimum_distance_gt = dist_mat.min(axis=0)
        return np.maximum(
            np.quantile(minimum_distance_pred, q),
            np.quantile(minimum_distance_gt, q),
        )

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
        if dist_mat.size == 0:
            return np.nan
        n_pred = sum(dist_mat.shape)
        minimum_distance = np.concatenate(
            [dist_mat.min(axis=1), dist_mat.min(axis=0)]
        )
        return np.sum(minimum_distance < max_distance) / n_pred

    def normalised_surface_distance(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        max_distance: float = 0.0,
    ) -> float:
        """
        Compute the normalised surface distance between the predicted and target
        images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.
            max_distance (float, optional): maximum distance. Defaults to 0.0.

        Returns:
            float: normalised surface distance.
        """

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
        if self.pred_preprocessing_fn is not None:
            pred = self.pred_preprocessing_fn(pred)
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
            yield pred_mask, gt_mask, i, j

        for j in range(n_gt):
            if j not in col_ind:
                gt_mask = gt_labels == (j + 1)
                pred_mask = np.zeros_like(pred)
                yield pred_mask, gt_mask, None, j

    def __center(self, array: np.ndarray) -> np.ndarray:
        """
        Computes the center of an object in an array.

        Args:
            array (np.ndarray): array.

        Returns:
            np.ndarray: center of the positive pixels in the array.
        """
        return [float(x) for x in ndimage.center_of_mass(array)]

    def calculate_case(
        self,
        pred: ImageMultiFormat,
        gt: ImageMultiFormat,
        metrics: list[str] = None,
        match_regions: bool = False,
    ) -> dict[str, float]:
        """
        Compute the metrics between the predicted and target images.

        Args:
            pred (ImageMultiFormat): predicted image.
            gt (ImageMultiFormat): target image.
            metrics (list[str]): list of metrics to compute.
            match_regions (bool): whether to match regions. Defaults to False.

        Returns:
            dict[str, float]: dictionary of metrics.
        """

        if metrics is None:
            if self.metrics is None:
                metrics = self.metric_match.keys()
            else:
                metrics = self.metrics
        output_dict = {k: [] for k in metrics}
        output_dict.update(
            {
                "idx_pred": [],
                "idx_gt": [],
                "center_pred": [],
                "center_gt": [],
                "pred_size": [],
                "gt_size": [],
            }
        )
        pred_gt_iterator = (
            self.__match_and_iterate_regions(pred, gt)
            if match_regions
            else [(pred, gt, 0, 0)]
        )
        for matched_pred, matched_gt, idx_pred, idx_gt in pred_gt_iterator:
            for metric in metrics:
                if metric in self.metric_match:
                    output_dict[metric].append(
                        self.metric_match[metric](
                            matched_pred, matched_gt, **self.params[metric]
                        )
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            output_dict["idx_pred"].append(int(idx_pred))
            output_dict["idx_gt"].append(int(idx_gt))
            output_dict["center_pred"].append(self.__center(matched_pred))
            output_dict["center_gt"].append(self.__center(matched_gt))
            output_dict["pred_size"].append(int(np.sum(matched_pred)))
            output_dict["gt_size"].append(int(np.sum(matched_gt)))
        return output_dict

    def calculate_metrics_standard(
        self,
        preds: list[ImageMultiFormat],
        gts: list[ImageMultiFormat],
        metric_ids: list[str],
    ) -> list[dict]:
        """
        Calculate metrics for multiple predictions and ground truth pairs and
        returns a structured dictionary with the output.

        Args:
            preds (list[ImageMultiFormat]): predictions.
            gts (list[ImageMultiFormat]): ground truths.
            metric_ids (list[str]): list of metrics to compute.

        Returns:
            list[dict]: metrics.

        Raises:
            ValueError: if preds and gts have different lengths.
        """
        if len(preds) != len(gts):
            raise ValueError("preds and gts must have the same length")
        output = []
        average_values = {metric: [] for metric in metric_ids}
        n = 0
        iterator = enumerate(zip(preds, gts))
        if self.verbose:
            iterator = tqdm(iterator, total=len(preds))
        for i, (pred, gt) in iterator:
            pred_path = pred if isinstance(pred, str) else str(i)
            gt_path = gt if isinstance(gt, str) else str(i)
            pred, gt = self.load_arrays(pred, gt)
            metrics = self.calculate_case(pred, gt, metric_ids)
            metrics["pred_path"] = pred_path
            metrics["gt_path"] = gt_path
            for metric in average_values:
                average_values[metric].extend(metrics[metric])
            n += 1
            output.append(metrics)
        output_dict = {"metrics": output}
        for k in self.aggregation_functions:
            for metric in average_values:
                average_values[metric] = np.array(average_values[metric])
                if metric not in output_dict:
                    output_dict[metric] = {}
                if self.n_classes > 2:
                    output_dict[metric][k] = {}
                    for cl in range(self.n_classes):
                        output_dict[metric][k][cl] = self.aggregation_functions[
                            k
                        ](average_values[metric][:, cl])
                else:
                    output_dict[metric][k] = {
                        1: self.aggregation_functions[k](average_values[metric])
                    }

        return output_dict

    def calculate_metrics_with_match_regions(
        self,
        preds: list[ImageMultiFormat],
        gts: list[ImageMultiFormat],
        metric_ids: list[str],
    ) -> list[dict]:
        """
        Calculate metrics for multiple predictions and ground truth pairs and
        returns a structured dictionary with the output.

        Args:
            preds (list[ImageMultiFormat]): predictions.
            gts (list[ImageMultiFormat]): ground truths.
            metric_ids (list[str]): list of metrics to compute.

        Returns:
            list[dict]: metrics.

        Raises:
            ValueError: if preds and gts have different lengths.
        """
        if len(preds) != len(gts):
            raise ValueError("preds and gts must have the same length")
        output = []
        average_values = {
            metric: {cl: [] for cl in range(self.n_classes)}
            for metric in metric_ids
        }
        n = 0
        if self.n_classes > 2:
            real_n_classes = self.n_classes
            cl_range = range(real_n_classes)
        else:
            real_n_classes = 1
            cl_range = [0]
        self.n_classes = 2
        iterator = enumerate(zip(preds, gts))
        if self.verbose:
            iterator = tqdm(iterator, total=len(preds))
        for i, (pred, gt) in iterator:
            pred_path = pred if isinstance(pred, str) else str(i)
            gt_path = gt if isinstance(gt, str) else str(i)
            pred, gt = self.load_arrays(
                pred,
                gt,
                convert_to_one_hot=real_n_classes > 1,
                n_classes=real_n_classes,
            )
            if real_n_classes == 1:
                # add a dimension to the predictions and ground truth if the
                # task is binary.
                pred = pred[None] if pred.shape[0] > 2 else pred
                gt = gt[None] if gt.shape[0] > 2 else gt
            case_metrics = []
            for cl in cl_range:
                metrics = self.calculate_case(
                    pred[cl], gt[cl], metric_ids, match_regions=True
                )
                metrics["pred_id"] = pred if isinstance(pred, str) else str(i)
                metrics["gt_id"] = gt if isinstance(gt, str) else str(i)
                metrics["class"] = cl
                for metric in average_values:
                    average_values[metric][cl].extend(metrics[metric])
                n += 1
                metrics["pred_path"] = pred_path
                metrics["gt_path"] = gt_path
                case_metrics.append(metrics)
            case_metrics = {
                k: [cm[k] for cm in case_metrics] for k in case_metrics[0]
            }
            output.append(case_metrics)
        output_dict = {"metrics": output}
        for k in self.aggregation_functions:
            for metric in average_values:
                if metric not in output_dict:
                    output_dict[metric] = {}
                if real_n_classes > 2:
                    output_dict[metric][k] = {
                        cl: self.aggregation_functions[k](
                            average_values[metric][cl], axis=-1
                        )
                        for cl in cl_range
                    }
                else:
                    output_dict[metric][k] = {
                        1: self.aggregation_functions[k](
                            average_values[metric][0]
                        )
                    }

        self.n_classes = real_n_classes
        return output_dict

    def calculate_metrics(
        self,
        preds: list[ImageMultiFormat],
        gts: list[ImageMultiFormat],
        metric_ids: list[str] | None = None,
        match_regions: bool | None = None,
    ) -> list[dict]:
        """
        Calculate metrics for multiple predictions and ground truth pairs and
        returns a structured dictionary with the output. Region matching (when
        predicted and ground truth connected components are matched and metrics
        are calculated only between matching regions) is supported.

        Args:
            preds (list[ImageMultiFormat]): predictions.
            gts (list[ImageMultiFormat]): ground truths.
            metric_ids (list[str] | None): metrics to calculate. Defaults to
                None.
            match_regions (bool, optional): if True, predicted and ground truth
                regions will be matched. Defaults to None (uses
                self.match_regions).

        Returns:
            list[dict]: metrics.
        """
        if metric_ids is None:
            if self.metrics is None:
                metric_ids = self.metric_match.keys()
            else:
                metric_ids = self.metrics
        match_regions = (
            self.match_regions if match_regions is None else match_regions
        )
        if match_regions:
            out = self.calculate_metrics_with_match_regions(
                preds, gts, metric_ids
            )
        else:
            out = self.calculate_metrics_standard(preds, gts, metric_ids)

        return coherce_to_non_array(out)
