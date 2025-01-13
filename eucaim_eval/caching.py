import hashlib
from functools import partial
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk


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
