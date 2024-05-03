import seaborn as sns
import matplotlib.pyplot as plt

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
    :return: Nothing returned, plot is shown
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
