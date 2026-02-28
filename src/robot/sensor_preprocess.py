import numpy as np
from collections import defaultdict
from typing import Tuple, Any, Dict


def estimate_bias_sigma(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-dimension bias (mu) and std (sigma) from samples.

    samples: (..., dim) array or (N,) for 1D
    Returns (mu, sigma)
    """
    arr = np.array(samples)
    if arr.ndim == 1:
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=0)
    else:
        mu = np.mean(arr, axis=0)
        sigma = np.std(arr, axis=0, ddof=0)
    return mu, sigma


def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / float(window)
    # for 2D arrays, apply per-column
    return np.vstack([moving_average(data[:, i], window) for i in range(data.shape[1])]).T


def exponential_moving_average(data: np.ndarray, alpha: float = 0.2, state: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute EMA. If state provided, continue from previous.
    Returns (ema_values, final_state)
    """
    data = np.asarray(data)
    if state is None:
        ema = data[0].astype(float)
    else:
        ema = np.array(state, dtype=float)

    out = []
    for x in data:
        ema = alpha * np.asarray(x) + (1 - alpha) * ema
        out.append(ema.copy())
    return np.array(out), ema


def median_filter(data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    data = np.asarray(data)
    if kernel_size <= 1:
        return data
    if data.ndim == 1:
        half = kernel_size // 2
        padded = np.pad(data, (half, half), mode='edge')
        out = [np.median(padded[i:i + kernel_size]) for i in range(len(data))]
        return np.array(out)
    # apply per-column
    return np.vstack([median_filter(data[:, i], kernel_size) for i in range(data.shape[1])]).T


class SimpleKalman1D:
    def __init__(self, process_var: float = 1e-5, meas_var: float = 1e-2):
        self.process_var = process_var
        self.meas_var = meas_var
        self.x = None
        self.P = None

    def reset(self, initial_x: float = 0.0, initial_P: float = 1.0):
        self.x = float(initial_x)
        self.P = float(initial_P)

    def filter(self, measurements: np.ndarray) -> np.ndarray:
        out = []
        for z in measurements:
            if self.x is None:
                self.x = float(z)
                self.P = 1.0
                out.append(self.x)
                continue
            # predict
            Ppred = self.P + self.process_var
            # update
            K = Ppred / (Ppred + self.meas_var)
            self.x = self.x + K * (float(z) - self.x)
            self.P = (1 - K) * Ppred
            out.append(self.x)
        return np.array(out)


class SensorPreprocessor:
    """Stateful preprocessor supporting EMA, moving average, median, and Kalman filtering."""

    def __init__(self):
        self.ema_state: Dict[Any, np.ndarray] = {}
        self.kalman_state: Dict[Any, SimpleKalman1D] = {}

    def calibrate(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return estimate_bias_sigma(samples)

    def preprocess(self, data: np.ndarray, method: str = "ema", key: Any = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess `data` (array of shape [T] or [T, D]).

        Returns (filtered_data, covariance_estimate)
        covariance_estimate is per-dimension variance (sigma^2)
        """
        arr = np.asarray(data)
        if method == "ema":
            alpha = kwargs.get("alpha", 0.2)
            prev = self.ema_state.get(key, None)
            filtered, final = exponential_moving_average(arr, alpha=alpha, state=prev)
            if key is not None:
                self.ema_state[key] = final
            # estimate covariance from residuals
            residuals = arr - filtered
            cov = np.var(residuals, axis=0) if residuals.ndim > 1 else np.var(residuals)
            return filtered, cov

        if method == "moving_average":
            window = kwargs.get("window", 5)
            filtered = moving_average(arr, window=window)
            residuals = arr[window - 1:] - filtered
            cov = np.var(residuals, axis=0) if residuals.ndim > 1 else np.var(residuals)
            return filtered, cov

        if method == "median":
            kernel = kwargs.get("kernel_size", 3)
            filtered = median_filter(arr, kernel_size=kernel)
            residuals = arr - filtered
            cov = np.var(residuals, axis=0) if residuals.ndim > 1 else np.var(residuals)
            return filtered, cov

        if method == "kalman":
            key = key or "default"
            kf = self.kalman_state.get(key)
            if kf is None:
                kf = SimpleKalman1D(process_var=kwargs.get("process_var", 1e-5), meas_var=kwargs.get("meas_var", 1e-2))
                self.kalman_state[key] = kf
            filtered = kf.filter(np.ravel(arr))
            # approximate covariance with last P
            cov = kf.P if kf.P is not None else 0.0
            return filtered, cov

        raise ValueError(f"Unknown preprocessing method: {method}")


def add_gaussian_noise(data: np.ndarray, mu=0.0, sigma=0.01) -> np.ndarray:
    return np.array(data) + np.random.normal(mu, sigma, size=np.shape(data))


__all__ = [
    "estimate_bias_sigma",
    "moving_average",
    "exponential_moving_average",
    "median_filter",
    "SimpleKalman1D",
    "SensorPreprocessor",
    "add_gaussian_noise",
]
