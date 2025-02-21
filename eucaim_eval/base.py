from abc import ABC
from dataclasses import dataclass
from multiprocessing import pool
from typing import Callable

import warnings
import numpy as np
import SimpleITK as sitk

from .bootstrap import Bootstrap

ImageMultiFormat = str | sitk.Image


@dataclass
class AbstractMetrics(ABC):
    """
    Abstract class for metrics.
    """

    n_classes: int = 2
    reduction: None | str | Callable = None
    input_is_one_hot: bool = False
    params: dict = None
    seed: int = 42
    ci: float = 0.95
    n_workers: int = 1
    metrics: list[str] = None
    cache_size: int = 0
    verbose: bool = False

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = self.AVAILABLE_METRICS
        if self.cache_size > 0:
            self.set_cache(self.cache_size)
        self.params = self.params or {}
        for metric in self.metric_match:
            if metric not in self.params:
                self.params[metric] = {}
        self.rng = np.random.default_rng(self.seed)
        self.q = (1 - 0.95) / 2
        self.q = self.q, 1 - self.q

    @property
    def metric_match(self, *args, **kwargs):
        raise NotImplementedError(
            f"metric_match not implemented "
            "({self.__name__} is an abstract class)."
        )

    def calculate_metrics(self, *args, **kwargs) -> dict:
        raise NotImplementedError(
            f"calculate_metrics not implemented "
            "({self.__name__} is an abstract class)."
        )

    def __ci(self, x: np.ndarray, *args, **kwargs):
        return np.nanquantile(x, self.q, *args, **kwargs)

    @property
    def aggregation_functions(self):
        return {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "std": np.nanstd,
            "min": np.nanmin,
            "max": np.nanmax,
            "ci_95": self.__ci,
        }

    def bootstrap(
        self,
        arrays: list[np.ndarray],
        fn: Callable,
        n_bootstraps: int = 1000,
        bootstrap_size: int | float = 0.5,
        mp_pool: pool.Pool = None,
    ):
        """
        Generic bootstrap function.

        Args:
            arrays (list[np.ndarray]): arrays which will be sampled.
            fn (Callable): function to apply to the arrays.
            n_bootstraps (int, optional): number of samples. Defaults to
                1000.
            bootstrap_size (int | float, optional): size of bootstrap samples.
                Defaults to 0.5.
            mp_pool (pool.Pool, optional): multiprocessing pool. Defaults to
                None.

        Returns:
            list[np.ndarray]: results of the bootstrap.
        """

        bootstrap = Bootstrap(
            arrays=arrays,
            fn=fn,
            n_bootstraps=n_bootstraps,
            bootstrap_size=bootstrap_size,
            rng=self.rng,
        )
        results = bootstrap.bootstrap(mp_pool=mp_pool)
        return results

    def to_one_hot(
        self, array: np.ndarray, n_classes: int | None = None
    ) -> np.ndarray:
        """
        Convert the array to one-hot encoding.

        Args:
            array (np.ndarray): array to convert.
            n_classes (int, optional): number of classes. Overrides
                self.n_classes. Defaults to None.

        Returns:
            np.ndarray: one-hot encoded array.
        """
        n_classes = self.n_classes if n_classes is None else n_classes
        oh = np.eye(n_classes)[array.astype(int)]
        if len(oh.shape) == 3:
            oh = oh.transpose([2, 0, 1])
        return oh

    def set_cache(self, maxsize: int = 1000):
        """
        Placeholder for set_cache method.
        """
        warnings.warn(f"cache set to {maxsize} but set_cache not implemented.")


@dataclass
class ImabeBasedMetrics(AbstractMetrics, ABC):
    """
    Class to compute classification metrics.
    """

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
        if len(images) > 1:
            for image in images[1:]:
                if image.GetSize() != images[0].GetSize():
                    raise ValueError("Images must have the same size.")
                if image.GetSpacing() != images[0].GetSpacing():
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

    def load_arrays(
        self,
        *images: list[ImageMultiFormat | np.ndarray],
        convert_to_one_hot: bool = False,
        n_classes: int | None = None,
    ) -> list[np.ndarray]:
        """
        Load the images as numpy arrays.

        Args:
            *images (ImageMultiFormat): images to load.
            convert_to_one_hot (bool, optional): forces conversion to one-hot.
                Defaults to False.
            n_classes (int, optional): number of classes. Defaults to None.

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
        if (self.input_is_one_hot is False) and (
            self.n_classes > 2 or convert_to_one_hot
        ):
            output = [self.to_one_hot(image, n_classes) for image in output]
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
