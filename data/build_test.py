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

    microp_left = open_image('../../../RU/data/Microplastics Sample #2/Microp2_Img1_2024-05-07_12-16-06/'
                             'capture/Microp2_Img1_2024-05-07_12-16-06.hdr')
    microp_mid = open_image('../../../RU/data/Microplastics Sample #2/Microp2_Img2_2024-05-07_12-20-04/'
                            'capture/Microp2_Img2_2024-05-07_12-20-04.hdr')
    microp_right = open_image('../../../RU/data/Microplastics Sample #2/Microp2_Img3_2024-05-07_12-23-50/'
                              'capture/Microp2_Img3_2024-05-07_12-23-50.hdr')

    microp_left_data = microp_left[1000:4300, :, :]
    microp_mid_data = microp_mid[1000:4300, :, :]
    microp_right_data = microp_right[1000:4300, :, :]

    microp_mid_shifted = np.roll(microp_mid_data[:, :, :], -85, axis=0)
    microp_mid_shifted_cropped = microp_mid_shifted[:, 20:, :]

    microp_right_shifted = np.roll(microp_right_data[:, :, :], -120, axis=0)
    microp_right_shifted_cropped = microp_right_shifted[:, 190:, :]

    imgs_to_merge = [microp_left_data, microp_mid_shifted_cropped, microp_right_shifted_cropped]
    merged = merge_images(imgs_to_merge, metadata=microp_left.metadata)

    merged = merged[:, :, 40:180]
    return merged


microp_img = construct_microplastics_test()
np.save('test_img_2', microp_img)
