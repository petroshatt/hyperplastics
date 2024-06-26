import numpy as np
import pybaselines
from scipy.signal import savgol_filter, convolve2d, find_peaks
from skimage.restoration import denoise_wavelet


def neighbouring_summation(array_3D, window_size=5):
    """
    Function that applies neighbouring summation for each pixel of the original three-dimensional array
    :param array_3D: Original three-dimensional array before sum
    :param window_size: The dimensions of the kernel window,
                        height and width are always the same, must be an odd number
    :return: Three-dimensional array after summing up the neighbors for each pixel
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number")

    kernel = np.ones((window_size, window_size))
    center = window_size // 2
    kernel[center, center] = 0

    output = np.zeros_like(array_3D)
    for c in range(array_3D.shape[2]):
        output[:, :, c] = convolve2d(array_3D[:, :, c], kernel, mode='same', boundary='wrap')
    return output


def peak_annonation_1D(array_1D):
    array_1D = log_als_1D(array_1D)
    array_1D = np.reshape(array_1D, 60)
    peaks, _ = find_peaks(array_1D, height=(0.3*np.amax(array_1D)))

    # plt.plot(array_1D)
    # plt.plot(peaks, array_1D[peaks], "x")
    # plt.plot(np.zeros_like(array_1D), "--", color="gray")
    # plt.show()

    output = []
    output.append(len(peaks))

    peak_intensities_sum = -1
    peak_distances_sum = -1
    relative_intensity = -1

    if len(peaks) == 1:
        peak_intensities_sum = array_1D[peaks[0]]
        peak_distances_sum = 0
        relative_intensity = 1
    elif len(peaks) == 2:
        peak_intensities_sum = (array_1D[peaks[0]] + array_1D[peaks[1]])
        peak_distances_sum = (peaks[1] - peaks[0])
        relative_intensity = (array_1D[peaks[0]] / array_1D[peaks[1]])

    output.append(peak_intensities_sum)
    output.append(peak_distances_sum)
    output.append(relative_intensity)
    return output


def peak_annotation(array_2D):
    return np.apply_along_axis(peak_annonation_1D, 1, array_2D)


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
    return np.apply_along_axis(log_als_1D, 1, array_2D).reshape(array_2D.shape)


def wavelet_denoise_1D(array_1D, wavelet_levels=6):
    """
    skimage Wavelet Denoising function
    :param array_1D: One-dimensional array to be transformed
    :param wavelet_levels: The number of wavelet decomposition levels to use
    :return: Transformed 1D array
    """
    return denoise_wavelet(array_1D, wavelet='db1', mode='soft', wavelet_levels=wavelet_levels)


def wavelet_denoise(array_2D):
    """
    wavelet_denoise applied to each row of 2D array
    :param array_2D: Two-dimensional array to be transformed
    :return: Transformed 2D array
    """
    return np.apply_along_axis(wavelet_denoise_1D, 1, array_2D)


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
