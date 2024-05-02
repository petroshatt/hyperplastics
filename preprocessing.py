import numpy as np
import pybaselines
from scipy.signal import savgol_filter, convolve2d
from skimage.restoration import denoise_wavelet


def neighbouring_summation(array_3D):
    """
    [NEEDS TO BE TESTED (Optimized Version) - If not working, replace with function from Neighbors224]
    Function that applies neighbouring summation for each pixel of the original three-dimensional array
    :param array_3D: Original three-dimensional array before sum
    :return: Three-dimensional array after summing up the neighbors for each pixel
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    output = np.zeros_like(array_3D)
    for c in range(array_3D.shape[2]):
        output[:, :, c] = convolve2d(array_3D[:, :, c], kernel, mode='same', boundary='wrap')
    return output


def savgol(array, window_length=5, polyorder=2, deriv=1):
    """
    Savitzky-Golay filter
    :param array: Array (can be one or two dimensions) for the filter to be applied
    :param window_length: Window length
    :param polyorder: Polynomial order
    :param deriv: Derivative (1st or 2nd)
    :return: Array after applying the filter
    """
    return savgol_filter(array, window_length=window_length, polyorder=polyorder, deriv=deriv)


def log_als_1D(array_1D):
    """
    Negative logarithm (-log) to get spectrum absorbance +
    Asymmetric Least Squares (ALS) for background correction
    :param array_1D: One-dimensional array to be transformed
    :return: Transformed 1D array
    """
    # Avoid taking the logarithm of zero or negative values
    array_1D = np.maximum(array_1D, 1e-10)

    x_log = np.reshape(-np.log(array_1D), (1, -1))
    x_baseline, _ = pybaselines.whittaker.asls(x_log, lam=10000)
    x_baseline = x_baseline.reshape(1, -1)

    output = x_log - x_baseline
    return output


def log_als(array_2D):
    """
    log_als applied to each row of 2D array
    :param array_2D: Two-dimensional array to be transformed
    :return: Transformed 2D array
    """
    return np.apply_along_axis(log_als, 1, array_2D)


def wavelet_denoise_1D(array_1D, wavelet_levels=6):
    return denoise_wavelet(array_1D, wavelet='db1', mode='soft', wavelet_levels=wavelet_levels)


def wavelet_denoise(array_2D):
    """
    wavelet_denoise applied to each row of 2D array
    :param array_2D: Two-dimensional array to be transformed
    :return: Transformed 2D array
    """
    return np.apply_along_axis(wavelet_denoise, 1, array_2D)


def min_max_normalization(array_2D):
    """
    Min-Max (0 to 1) Normalization for each row of 2D array
    :param array_2D: Two-dimensional array to be transformed
    :return: Transformed 2D array
    """
    min_vals = np.min(array_2D, axis=1, keepdims=True)
    max_vals = np.max(array_2D, axis=1, keepdims=True)
    return (array_2D - min_vals) / (max_vals - min_vals)


def mean_normalization(array_2D):
    """
    Mean Normalization for each row of 2D array
    :param array_2D: Two-dimensional array to be transformed
    :return: Transformed 2D array
    """
    mean_vals = np.mean(array_2D, axis=1, keepdims=True)
    return array_2D - mean_vals


def area_normalization(array_2D):
    """
    Area Normalization for each row of 2D array
    :param array_2D: Two-dimensional array to be transformed
    :return: Transformed 2D array
    """
    areas = np.sum(array_2D, axis=1, keepdims=True)
    return array_2D / areas
