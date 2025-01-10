import numpy as np
import SimpleITK as sitk
from abc import ABC
from typing import Callable
from dataclasses import dataclass

ImageMultiFormat = str | sitk.Image


@dataclass
class AbstractMetrics(ABC):
    """
    Abstract class for metrics.
    """

    n_classes: int = 2
    reduction: None | str | Callable = None
    is_input_one_hot: bool = False
    params: dict = None
    seed: int = 42

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
        fn: Callable,
        n_bootstraps: int = 1000,
        bootstrap_size: int | float = 0.5,
        *arrays,
    ):
        """
        Generic bootstrap function.

        Args:
            fn (Callable): function to apply to the arrays.
            n_bootstraps (int, optional): number of samples. Defaults to
                1000.
            bootstrap_size (int | float, optional): size of bootstrap samples.
                Defaults to 0.5.
            *arrays (list[np.ndarray]): arrays which will be sampled.

        Returns:
            list[np.ndarray]: results of the bootstrap.
        """
        if isinstance(bootstrap_size, float):
            bootstrap_size = int(bootstrap_size * len(arrays[0]))
        results = [
            fn(
                *[
                    self.rng.choice(array, replace=False, size=bootstrap_size)
                    for array in arrays
                ]
            )
            for _ in range(n_bootstraps)
        ]
        return results


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

    def to_one_hot(self, image: np.ndarray) -> np.ndarray:
        """
        Convert the image to one-hot encoding.

        Args:
            image (np.ndarray): Image to convert.

        Returns:
            np.ndarray: One-hot encoded image.
        """
        return np.eye(self.n_classes)[image.astype(int)]

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
