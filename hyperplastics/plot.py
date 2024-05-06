import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
Matplotlib styles used are created by Dominik Haitz
'''
# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')


def plot_spectra(array_2D, wavelengths, title='Spectra Plot'):
    """
    Plot spectra of all row samples in a 2D array [X-axis: Wavalengths, Y-axis: Intensity]
    :param array_2D: The 2D array contaning the spectra to be plotted
    :param wavelengths: The wavelengths of the spectra
    :param title: Plot title
    :return: Nothing returned, spectra plot is shown
    """
    wavelengths = list(map(float, wavelengths))
    samples = range(1, array_2D.shape[0] + 1)

    plt.figure(figsize=(16, 5))

    for i, sample in enumerate(samples):
        intensity_values = array_2D[i]
        plt.plot(wavelengths, intensity_values, linewidth=2, label=str(sample))

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(str(title))
    plt.grid(True)

    plt.show()


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


def plot_predictions(y_pred, initial_shape):
    """
    Plot predictions of the model, reshaping 1D to 2D and assigning a color to each class
    :param y_pred: The predictions array returned by the model
    :param initial_shape: The initial shape of the image so the predictions can be reshaped and plotted in 2D
    :return: Nothing returned, predictions plot is shown
    """
    class_array = np.reshape(y_pred, (initial_shape[0], initial_shape[1]))
    class_colors = {'PP': 'blue', 'PVC': 'green', 'PE': 'orange', 'PET': 'purple'}

    plt.figure(figsize=(12, 12))

    for i in range(class_array.shape[0]):
        for j in range(class_array.shape[1]):
            if class_array[i, j] != 'UNCL':
                plt.scatter(j, i, c=class_colors[class_array[i, j]])

    # Reverse y-axis
    plt.gca().invert_yaxis()

    # Create legend manually
    legend_handles = []
    for class_label, color in class_colors.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                         markersize=10, label=class_label))

    plt.legend(handles=legend_handles)
    plt.show()
