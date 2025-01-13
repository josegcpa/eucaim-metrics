from dataclasses import dataclass
from multiprocessing import pool
from typing import Callable

import numpy as np


@dataclass
class Bootstrap:
    """
    Class to compute bootstrap.
    """

    arrays: list[np.ndarray]
    fn: Callable
    rng: np.random.Generator = np.random.default_rng()
    bootstrap_size: int | float = 0.5
    n_bootstraps: int = 1000

    def __post_init__(self):
        self.bootstrap_size = (
            int(self.arrays[0].shape[0] * self.bootstrap_size)
            if self.bootstrap_size < 1
            else self.bootstrap_size
        )

    def sample_arrays(
        self, *arrays: list[np.ndarray], seed: int = 42
    ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(
            arrays[0].shape[0], size=self.bootstrap_size, replace=False
        )
        return [array[idxs] for array in arrays]

    def sample_and_apply(self, seed: int) -> np.ndarray:
        return self.fn(*self.sample_arrays(*self.arrays, seed=seed))

    def bootstrap(self, mp_pool: pool.Pool = None):
        """
        Generic bootstrap function.

        Args:
            mp_pool (pool.Pool, optional): multiprocessing pool. Defaults to
                None.

        Returns:
            list[np.ndarray]: results of the bootstrap.
        """

        results = []
        seed_list = self.rng.integers(0, 1e6, self.n_bootstraps)
        map_fn = mp_pool.imap if mp_pool else map
        for result in map_fn(self.sample_and_apply, seed_list):
            results.append(result)
        return results
