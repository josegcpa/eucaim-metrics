import numpy as np


def coherce_to_non_array(
    data: dict | list | tuple | np.ndarray,
) -> dict | list | tuple:
    if isinstance(data, dict):
        for k in data:
            data[k] = coherce_to_non_array(data[k])
    elif isinstance(data, list):
        data = [coherce_to_non_array(d) for d in data]
    elif isinstance(data, tuple):
        data = tuple([coherce_to_non_array(d) for d in data])
    elif isinstance(data, np.ndarray):
        data = coherce_to_non_array(data.tolist())
    elif isinstance(data, (np.float32, np.float64)):
        data = float(data)
    return data
