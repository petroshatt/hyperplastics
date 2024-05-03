from spectral import *
import numpy as np


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
    microp_left = open_image('../../../RU/data/Microplastics_NoBG_Circle_224/Microplastics1_NoBG_2024-03-22_12-35-19/'
                             'capture/Microplastics1_NoBG_2024-03-22_12-35-19.hdr')
    microp_mid = open_image('../../../RU/data/Microplastics_NoBG_Circle_224/Microplastics2_NoBG_2024-03-22_12-37-54/'
                            'capture/Microplastics2_NoBG_2024-03-22_12-37-54.hdr')
    microp_right = open_image('../../../RU/data/Microplastics_NoBG_Circle_224/Microplastics4_NoBG_2024-03-22_12-45-07/'
                              'capture/Microplastics4_NoBG_2024-03-22_12-45-07.hdr')

    microp_left_data = microp_left[:, :, :]
    microp_mid_data = microp_mid[:, :, :]
    microp_right_data = microp_right[:, :, :]

    microp_mid_shifted = np.roll(microp_mid_data[:, :, :], -45, axis=0)
    microp_mid_shifted_cropped = microp_mid_shifted[:, 105:-65, :]

    microp_right_shifted = np.roll(microp_right_data[:, :, :], -85, axis=0)

    imgs_to_merge = [microp_left_data, microp_mid_shifted_cropped, microp_right_shifted]
    merged = merge_images(imgs_to_merge, metadata=microp_left.metadata)

    merged = merged[400:1750, :, 40:180]
    return merged


microp_img = construct_microplastics_test()
np.save('X_test', microp_img)
