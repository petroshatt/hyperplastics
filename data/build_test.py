from matplotlib import pyplot as plt
from spectral import *
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

def plot_mean_image(array_3D):
    """
    Plot mean image of a 3D image calculating the mean spectra for every pixel
    :param array_3D: 3D array to be plotted
    :return: Nothing returned, mean image plot is shown
    """
    data_means = np.mean(array_3D, axis=2)

    plt.figure(figsize=(16, 10))

    plt.imshow(data_means, interpolation='nearest')
    plt.title('Mean Image')
    plt.colorbar()

    plt.show()


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

    microp_left = open_image('../../../RU/data/6_1/NET1_2E_6_1/NET1_2E_6_1_A_2024-05-24_09-20-15/capture/'
                             'NET1_2E_6_1_A_2024-05-24_09-20-15.hdr')
    microp_mid = open_image('../../../RU/data/6_1/NET1_2E_6_1/NET1_2E_6_1_B_2024-05-24_09-23-08/capture/'
                            'NET1_2E_6_1_B_2024-05-24_09-23-08.hdr')
    microp_right = open_image('../../../RU/data/6_1/NET1_2E_6_1/NET1_2E_6_1_C_2024-05-24_09-28-59/capture/'
                              'NET1_2E_6_1_C_2024-05-24_09-28-59.hdr')

    microp_left_data = microp_left[:, :, :]
    microp_mid_data = microp_mid[:, :, :]
    microp_right_data = microp_right[:, :, :]

    microp_mid_shifted = np.roll(microp_mid_data[:, :, :], -18, axis=0)
    microp_mid_shifted_cropped = microp_mid_shifted[:, 30:, :]

    microp_right_shifted = np.roll(microp_right_data[:, :, :], -38, axis=0)
    microp_right_shifted_cropped = microp_right_shifted[:, 235:, :]

    imgs_to_merge = [microp_left_data, microp_mid_shifted_cropped, microp_right_shifted_cropped]
    merged = merge_images(imgs_to_merge, metadata=microp_left.metadata)

    merged = merged[240:2820, :, 40:180]
    return merged


microp_img = construct_microplastics_test()
np.save('6_1/NET1_6_1', microp_img)
