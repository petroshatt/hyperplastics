# import matplotlib
# matplotlib.use("TkAgg")
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

    plt.figure(figsize=(16, 3))

    for i, sample in enumerate(samples):
        intensity_values = array_2D[i]
        plt.plot(wavelengths, intensity_values, linewidth=2, label=str(sample))

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(str(title))
    plt.grid(True)
    # plt.legend()
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
    plt.tight_layout()
    # plt.colorbar()

    plt.show()


def plot_predictions(y_pred, initial_shape, title='Predictions Plot'):
    """
    Plot predictions of the model, reshaping 1D to 2D and assigning a color to each class
    :param y_pred: The predictions array returned by the model
    :param initial_shape: The initial shape of the image so the predictions can be reshaped and plotted in 2D
    :param title: Plot title
    :return: Nothing returned, predictions plot is shown
    """
    class_array = np.reshape(y_pred, initial_shape[:2])
    # class_colors = {'PP': '#6878c0', 'PVC': '#79b791', 'PE': '#ff9d5c', 'PET': '#b580c5', 'PS': '#40e0d0'}
    class_colors = {'PP': '#6878c0', 'PE': '#ff9d5c', 'PET': '#b580c5', 'PS': '#40e0d0'}

    # class_array[:25, :80] = 'UNCL'
    # class_array[100:, :80] = 'UNCL'
    # class_array[:25, 100:] = 'UNCL'
    # class_array[100:, 100:] = 'UNCL'

    fig, ax = plt.subplots(figsize=(17, ((17*initial_shape[0]) / initial_shape[1])))

    mask = (class_array != 'UNCL')
    rows, cols = np.where(mask)
    colors = np.array([class_colors[class_array[r, c]] for r, c in zip(rows, cols)])
    scatter = ax.scatter(cols, rows, c=colors)

    plt.xlim(0, initial_shape[1])
    plt.ylim(0, initial_shape[0])
    plt.gca().invert_yaxis()

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                 markersize=15, label=class_label) for class_label, color in class_colors.items()]

    plt.title(str(title))
    plt.legend(handles=legend_handles)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(event):
        if event.inaxes == ax:
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    pos = scatter.get_offsets()[ind["ind"][0]]
                    annot.xy = pos
                    text = f"x: {cols[ind['ind'][0]]}, y: {rows[ind['ind'][0]]}"
                    annot.set_text(text)
                    annot.get_bbox_patch().set_facecolor(colors[ind["ind"][0]])
                    annot.get_bbox_patch().set_alpha(0.4)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", update_annot)
    plt.pause(0.01)
    plt.show()
