"""
Specifies preprocessing classes.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BinarizeProbabilities:
    """
    Binarize a binary segmentation.

    Args:
        image (np.ndarray): image to binarize.
        threshold (float, optional): threshold. Defaults to 0.5.

    Returns:
        np.ndarray: binarized image.
    """

    threshold: float = 0.5
    probability_index: int = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.threshold is not None:
            if self.probability_index is not None:
                image = image[self.probability_index]
            return (image > self.threshold).astype(np.uint8)
        return image


__all__ = ["BinarizeProbabilities"]
