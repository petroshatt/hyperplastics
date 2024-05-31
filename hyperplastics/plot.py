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
    Plot predictions of the model, reshaping 1D to 2D and assigning a color to each class.

    :param y_pred: The predictions array returned by the model.
    :param initial_shape: The initial shape of the image so the predictions can be reshaped and plotted in 2D.
    :param title: Plot title.
    :return: Nothing returned, predictions plot is shown.
    """
    class_array = np.reshape(y_pred, initial_shape[:2])
    class_colors = {'PP': '#6878c0', 'PE': '#ff9d5c', 'PET': '#b580c5', 'PS': '#40e0d0'}

    fig, ax = plt.subplots(figsize=(17, ((17 * initial_shape[0]) / initial_shape[1])))

    mask = (class_array != 'UNCL')
    rows, cols = np.where(mask)
    colors = np.array([class_colors[class_array[r, c]] for r, c in zip(rows, cols)])
    scatter = ax.scatter(cols, rows, c=colors, s=72)  # Increased marker size

    plt.xlim(0, initial_shape[1])
    plt.ylim(0, initial_shape[0])
    plt.gca().invert_yaxis()

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                 markersize=25, label=class_label) for class_label, color in
                      class_colors.items()]  # Increased marker size in legend

    plt.title(str(title), fontsize=32)
    plt.legend(handles=legend_handles, fontsize=25)  # Increased legend font size

    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)

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


def plot_predictions_on_mean(array_3D, y_pred, initial_shape, title='Predictions Plot'):
    """
    Plot mean image of a 3D image and overlay the predictions scatter plot on top.
    :param array_3D: 3D array to be plotted as the mean image.
    :param y_pred: The predictions array returned by the model.
    :param initial_shape: The initial shape of the image so the predictions can be reshaped and plotted in 2D.
    :param title: Plot title.
    :return: Nothing returned, combined plot is shown.
    """
    data_means = np.mean(array_3D, axis=2)

    class_array = np.reshape(y_pred, initial_shape[:2])
    class_colors = {'PP': '#6878c0', 'PE': '#ff9d5c', 'PET': '#b580c5', 'PS': '#40e0d0'}

    fig, ax = plt.subplots(figsize=(17, ((17 * initial_shape[0]) / initial_shape[1])))

    ax.imshow(data_means, interpolation='nearest', aspect='auto')

    mask = (class_array != 'UNCL')
    rows, cols = np.where(mask)
    colors = np.array([class_colors[class_array[r, c]] for r, c in zip(rows, cols)])

    scatter = ax.scatter(cols, rows, c=colors, s=72, alpha=0.6)  # 60% opacity

    plt.gca().invert_yaxis()

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                 markersize=25, label=class_label) for class_label, color in class_colors.items()]

    plt.title(str(title), fontsize=32)
    plt.legend(handles=legend_handles, fontsize=25)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)

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


def plot_pie(y_pred, title='Percentage of Each Class in Predictions'):
    """
    Plot pie chart that shows percentage of each class in the predictions.

    :param y_pred: The predictions array returned by the model.
    :param title: Plot title.
    :return: Nothing returned, pie chart plot is shown.
    """
    class_colors = {'PP': '#6878c0', 'PE': '#ff9d5c', 'PET': '#b580c5', 'PS': '#40e0d0', 'UNCL': '#000000'}

    unique, counts = np.unique(y_pred, return_counts=True)
    class_counts = dict(zip(unique, counts))

    total_counts = np.sum(counts)
    percentages = {cls: (count / total_counts) * 100 for cls, count in class_counts.items()}

    unclassified_count = class_counts.get('UNCL', 0)
    classified_count = total_counts - unclassified_count
    other_vs_polymers_labels = ['Other', 'Polymers']
    other_vs_polymers_counts = [unclassified_count, classified_count]
    other_vs_polymers_colors = ['#555555', '#acd8a7']

    detailed_labels = [cls for cls in unique if cls != 'UNCL']
    detailed_counts = [class_counts[cls] for cls in detailed_labels]
    detailed_percentages = [percentages[cls] for cls in detailed_labels]
    detailed_colors = [class_colors[cls] for cls in detailed_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.pie(other_vs_polymers_counts, labels=other_vs_polymers_labels, colors=other_vs_polymers_colors,
            autopct='%1.1f%%', startangle=140)
    ax1.set_title('Polymers vs. Other')

    ax2.pie(detailed_percentages, labels=detailed_labels, colors=detailed_colors, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Percentage of Each Class in Predictions')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
