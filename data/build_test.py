from matplotlib import pyplot as plt
from spectral import *
import numpy as np

from hyperplastics.plot import plot_mean_image

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')


class MergedImage(np.ndarray):
    """
    Merged image class that inherits ndarray but adds metadata to it
    so the HSI image metadata can be stored
    """
    def __new__(cls, input_img, metadata=None):
        obj = np.asarray(input_img).view(cls)
        obj.metadata = metadata
        return obj


def merge_images(imgs, metadata):
    dimensions = [img.shape[0] for img in imgs]
    min_dimension = min(dimensions)
    result = np.concatenate([img[:min_dimension, :] for img in imgs], axis=1)
    merged_result = MergedImage(result, metadata=metadata)
    return merged_result


def construct_microplastics_test():
    """
    Function specific for merging our three microplastics HS images into a single array
    :return: The merged array
    """

    microp_left = open_image('/Users/petros/hsi-data/env-dept-samples/6-1-2021/2E/NET3_2E_6_1/'
                             'NET3_2E_6_1_a_2024-05-31_11-39-26/capture/NET3_2E_6_1_a_2024-05-31_11-39-26.hdr')
    microp_mid = open_image('/Users/petros/hsi-data/env-dept-samples/6-1-2021/2E/NET3_2E_6_1/'
                            'NET3_2E_6_1_b_2024-05-31_11-42-33/capture/NET3_2E_6_1_b_2024-05-31_11-42-33.hdr')
    microp_right = open_image('/Users/petros/hsi-data/env-dept-samples/6-1-2021/2E/NET3_2E_6_1/'
                              'NET3_2E_6_1_c_2024-05-31_11-45-44/capture/NET3_2E_6_1_c_2024-05-31_11-45-44.hdr')

    microp_left_data = microp_left[:, :, :]
    microp_mid_data = microp_mid[:, :, :]
    microp_right_data = microp_right[:, :, :]

    # microp_left_shifted = np.roll(microp_left_data[:, :, :], -25, axis=0)

    microp_mid_shifted = np.roll(microp_mid_data[:, :, :], -12, axis=0)
    microp_mid_shifted_cropped = microp_mid_shifted[:, 110:, :]

    # microp_right_shifted = np.roll(microp_right_data[:, :, :], -25, axis=0)
    microp_right_shifted_cropped = microp_right_data[:, 115:, :]

    # microp_right_shifted_cropped = np.roll(microp_right_cropped[:, :, :], -87, axis=1)
    # microp_right_shifted_cropped[:, -87:, :] = microp_mid_shifted_cropped[10, 10, :]

    imgs_to_merge = [microp_left_data, microp_mid_shifted_cropped, microp_right_shifted_cropped]
    merged = merge_images(imgs_to_merge, metadata=microp_left.metadata)

    merged = merged[200:-274, :1625, 40:180]
    return merged


microp_img = construct_microplastics_test()
# print(microp_img.shape)
# plot_mean_image(microp_img)
np.save('6_1/2E/NET3_2E_6_1', microp_img)
