import numpy as np


def reshape_3D_to_2D(array_3D):
    """
    Reshape a 3D (x,y,λ) array into 2D (x*y,λ)
    :param array_3D: The 3D array to be reshaped
    :return: The reshaped 2D array
    """
    return np.reshape(array_3D, (array_3D.shape[0] * array_3D.shape[1], array_3D.shape[2]))


def reshape_2D_to_3D(array_2D, initial_shape):
    """
    Reshape a 2D (x*y,λ) array into 3D (x,y,λ)
    :param array_2D: The 3D array to be reshaped
    :param initial_shape: A tuple containing x and y. They need to be specified in order to reshape!
    :return: The reshaped 2D array
    """
    return np.reshape(array_2D, initial_shape)
