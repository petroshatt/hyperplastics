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


def get_plot_title(filepath):
    """
    Returns the title of the plot based on the file path
    (Example: 'data/5_1/1E/NET1_5_1.npy' to 'NET 1 from 5/1')
    :param filepath: Filepath of the test image
    :return: A string containing the title of the plot in format
    """
    return f"{filepath.split('/')[3].split('_')[0]}"f" from {filepath.split('/')[1].replace('_', '/')}"
