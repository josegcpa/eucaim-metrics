import numpy as np
import hashlib
import inspect
import SimpleITK as sitk
from functools import partial
from .bootstrap import Bootstrap
from abc import ABC
from typing import Any, Callable
from dataclasses import dataclass
from multiprocessing import pool

ImageMultiFormat = str | sitk.Image


class MethodCache:
    """
    Caches the output of a method. If the input is a numpy array, it first
    converts this using frozenset adn then stores it.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.maxsize = None

        self.cache = {}
        self.history = []

    def __get__(self, instance, owner):
        return partial(self.__call__, instance)

    def set_maxsize(self, maxsize: int | None):
        self.maxsize = maxsize

    def to_hash_if_necessary(self, obj: Any) -> str:
        """
        Converts unhashable types to hashes.

        Args:
            obj (Any): object to convert.

        Returns:
            str: hash of the object.
        """
        if isinstance(obj, np.ndarray):
            return hashlib.sha256(obj.tobytes()).hexdigest()
        elif isinstance(obj, sitk.Image):
            return hashlib.sha256(
                sitk.GetArrayFromImage(obj).tobytes()
            ).hexdigest()
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self.to_hash_if_necessary(item)
            return "".join(obj)
        else:
            return obj

    def __call__(self, *args, **kwargs):
        parent_class, args = args[0], args[1:]
        key = (
            tuple([self.to_hash_if_necessary(arg) for arg in args]),
            frozenset(kwargs.items()),
        )
        if key in self.cache:
            return self.cache[key]
        else:
            self.history.append(key)
            if self.maxsize is not None:
                if len(self.cache) >= self.maxsize:
                    self.cache.pop(self.history.pop(0))
            value = self.func(parent_class, *args, **kwargs)
            self.cache[key] = value
            return value


def cache(maxsize: int | None = None):
    """
    Decorator to cache the output of a method. Uses MethodCache to achieve
    this and sets the maxsize to the given value.
    """

    def wraper(func: Callable):
        cache = MethodCache(func)
        cache.set_maxsize(maxsize)
        return cache

    return wraper


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
    n_workers: int = 1

    def __post_init__(self):
        self.params = self.params or {}
        for metric in self.metric_match:
            if metric not in self.params:
                self.params[metric] = {}
        self.rng = np.random.default_rng(self.seed)

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

    def to_one_hot(self, array: np.ndarray) -> np.ndarray:
        """
        Convert the array to one-hot encoding.

        Args:
            array (np.ndarray): array to convert.

        Returns:
            np.ndarray: one-hot encoded array.
        """
        return np.eye(self.n_classes)[array.astype(int)]


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

    def load_arrays(
        self, *images: list[ImageMultiFormat | np.ndarray]
    ) -> list[np.ndarray]:
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
        if (self.input_is_one_hot is False) and (self.n_classes > 2):
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
