import numpy as np
from spectral import *


def construct_uncl_training_set():
    """
    Build Unclassified training set
    :return: An ndarray with all samples selected for UNCL training class
    """
    uncl_img1 = open_image(
        '../../../RU/data/meso_2024-04-12_08-32-22/capture/meso_2024-04-12_08-32-22.hdr')
    uncl1 = uncl_img1[:, :, :]

    uncl_areas = [uncl1[260:290, 290:320, 40:180], uncl1[145:180, 160:200, 40:180], uncl1[150:180, 495:520, 40:180],
                uncl1[730:760, 495:515, 40:180], uncl1[105:125, 290:385, 40:180], uncl1[620:665, 410:430, 40:180]]

    reshaped_areas = [np.reshape(area, (area.shape[0] * area.shape[1], -1)) for area in uncl_areas]
    reshaped_uncl = np.concatenate(reshaped_areas, axis=0)

    reshaped_uncl = remove_low_intensity(reshaped_uncl, 900)
    reshaped_uncl = remove_high_intensity(reshaped_uncl, 2000)

    uncl_training = reshaped_uncl
    return uncl_training


def construct_pp_training_set():
    """
    Build PP training set
    :return: An ndarray with all samples selected for PP training class
    """
    pp_img1 = open_image(
        '../../../RU/data/River_Plastics_26032024/PP_film_2024-03-26_07-34-06/capture/PP_film_2024-03-26_07-34-06.hdr')
    pp1 = pp_img1[:, :, :]

    pp_areas = [pp1[830:880, 40:100, 40:180], pp1[850:920, 220:350, 40:180], pp1[830:860, 540:570, 40:180],
                pp1[624:645, 490:540, 40:180]]

    reshaped_areas = [np.reshape(area, (area.shape[0] * area.shape[1], -1)) for area in pp_areas]
    reshaped_pp = np.concatenate(reshaped_areas, axis=0)

    reshaped_pp = remove_low_intensity(reshaped_pp, 900)
    reshaped_pp = remove_high_intensity(reshaped_pp, 2000)

    pp_training = reshaped_pp
    return pp_training


def construct_pe_training_set():
    """
    Build PE training set
    :return: An ndarray with all samples selected for PE training class
    """
    pe_img1 = open_image(
        '../../../RU/data/River_Plastics_26032024/PE_film_2024-03-26_07-30-16/capture/PE_film_2024-03-26_07-30-16.hdr')
    pe1 = pe_img1[:, :, :]

    pe_areas = [pe1[910:1010, 240:320, 40:180], pe1[1180:1240, 440:490, 40:180],
                pe1[620:670, 430:480, 40:180], pe1[1220:1290, 80:140, 40:180]]

    reshaped_areas = [np.reshape(area, (area.shape[0] * area.shape[1], -1)) for area in pe_areas]
    reshaped_pe = np.concatenate(reshaped_areas, axis=0)

    reshaped_pe = remove_low_intensity(reshaped_pe, 900)

    pe_training = reshaped_pe
    return pe_training


def construct_pet_training_set():
    """
    Build PET training set
    :return: An ndarray with all samples selected for PET training class
    """
    pet_img1 = open_image(
        '../../../RU/data/Plastic Types - 224 Training Set/PET1_2024-03-22_10-14-10/capture/PET1_2024-03-22_10-14-10.hdr')
    pet1 = pet_img1[:, :, :]

    pet_areas = [pet1[830:890, 290:350, 40:180], pet1[1140:1180, 380:490, 40:180]]

    reshaped_areas = [np.reshape(area, (area.shape[0] * area.shape[1], -1)) for area in pet_areas]
    reshaped_pet = np.concatenate(reshaped_areas, axis=0)

    reshaped_pet = remove_low_intensity(reshaped_pet, 900)

    rand_pet_areas = get_random_pixels(pet1, 3500)
    rand_pet_areas = rand_pet_areas[:, 40:180]

    pet_training = np.concatenate([reshaped_pet, rand_pet_areas])
    return pet_training


def construct_pvc_training_set():
    """
    Build PVC training set
    :return: An ndarray with all samples selected for PVC training class
    """
    pvc_img1 = open_image(
        '../../../RU/data/Plastic Types - 224 Training Set/PVC1_2024-03-22_10-48-29/capture/PVC1_2024-03-22_10-48-29.hdr')
    pvc1 = pvc_img1[:, :, :]

    pvc_areas = [pvc1[1300:1370, 140:200, 40:180], pvc1[860:930, 320:380, 40:180],
                 pvc1[1800:1860, 220:250, 40:180], pvc1[580:630, 140:170, 40:180]]

    reshaped_areas = [np.reshape(area, (area.shape[0] * area.shape[1], -1)) for area in pvc_areas]
    reshaped_pvc = np.concatenate(reshaped_areas, axis=0)

    reshaped_pvc = remove_low_intensity(reshaped_pvc, 1300)
    reshaped_pvc = remove_high_intensity(reshaped_pvc, 2200)

    rand_pvc_areas = get_random_pixels(pvc1, 3500)
    rand_pvc_areas = rand_pvc_areas[:, 40:180]

    pvc_training = np.concatenate([reshaped_pvc, rand_pvc_areas])
    return pvc_training


def construct_Xs(classes=None, shuffle=False):
    """
    Constructs Xs of the training set
    :param classes: A list with strings of classes to include in training
    :param shuffle: If true, shuffles the training arrays before neighbouring summation
    :return: A list with Xs of all the classes specified
    """
    Xs = []
    if classes is None:
        classes = ['PP', 'PE', 'PET', 'PVC']

    if 'UNCL' in classes:
        X_uncl = construct_uncl_training_set()
        X_uncl = remove_high_intensity(X_uncl, 2100)
        X_uncl = remove_low_intensity(X_uncl, 300)
        X_uncl = sum_training(X_uncl, shuffle=shuffle)
        Xs.append(X_uncl)
        print("UNCL Training Set shape: ", X_uncl.shape)

    if 'PP' in classes:
        X_pp = construct_pp_training_set()
        X_pp = remove_high_intensity(X_pp, 2100)
        X_pp = remove_low_intensity(X_pp, 300)
        X_pp = sum_training(X_pp, shuffle=shuffle)
        Xs.append(X_pp)
        print("PP Training Set shape: ", X_pp.shape)

    if 'PE' in classes:
        X_pe = construct_pe_training_set()
        X_pe = remove_high_intensity(X_pe, 2200)
        X_pe = remove_low_intensity(X_pe, 300)
        X_pe = sum_training(X_pe, shuffle=shuffle)
        Xs.append(X_pe)
        print("PE Training Set shape: ", X_pe.shape)

    if 'PET' in classes:
        X_pet = construct_pet_training_set()
        X_pet = remove_low_intensity(X_pet, 300)
        X_pet = sum_training(X_pet, shuffle=shuffle)
        Xs.append(X_pet)
        print("PET Training Set shape: ", X_pet.shape)

    if 'PVC' in classes:
        X_pvc = construct_pvc_training_set()
        X_pvc = remove_low_intensity(X_pvc, 300)
        X_pvc = sum_training(X_pvc, shuffle=shuffle)
        Xs.append(X_pvc)
        print("PVC Training Set shape: ", X_pvc.shape)

    if 'PS' in classes:
        pass

    return Xs


def construct_ys(Xs):
    """
    Constructs ys of the training set
    :param Xs: The list construct_Xs returned, containing Xs of the training set
    :return: A list with ys corresponding to the Xs of the training set
    """
    ys = []
    for X in Xs:
        if X.shape[0] > 0:
            y = np.full(X.shape[0], X[0, -1]).reshape(-1, 1)
            ys.append(y)
    return ys


def get_class_limits(Xs, classes=None):
    """
    Returns the indices of the classes in the training set
    :param Xs: The list construct_Xs returned, containing Xs of the training set
    :param classes: A list with strings of classes to include in training
    :return: A dict with classes names as keys and tuples with starting and ending indices as values
    """
    if classes is None:
        classes = ['PP', 'PE', 'PET', 'PVC']

    class_limits = {}
    start_index = 0
    for class_name, array in zip(classes, Xs):
        end_index = start_index + array.shape[0]
        class_limits[class_name] = (start_index, end_index)
        start_index = end_index

    return class_limits


def remove_low_intensity(array_2D, threshold):
    """
    Removes low intensity samples from a 2D array, calculating the mean intensity of each row
    :param array_2D: The 2D array with the samples
    :param threshold: The specified threshold to remove each sample that has a mean intensity below this
    :return: The 2D array with removed samples
    """
    row_means = np.mean(array_2D, axis=1)
    indices_to_remove = np.where(row_means < threshold)[0]
    filtered_data = np.delete(array_2D, indices_to_remove, axis=0)
    return filtered_data


def remove_high_intensity(array_2D, threshold):
    """
    Removes high intensity samples from a 2D array, calculating the mean intensity of each row
    :param array_2D: The 2D array with the samples
    :param threshold: The specified threshold to remove each sample that has a mean intensity above this
    :return: The 2D array with removed samples
    """
    row_means = np.mean(array_2D, axis=1)
    indices_to_remove = np.where(row_means > threshold)[0]
    filtered_data = np.delete(array_2D, indices_to_remove, axis=0)
    return filtered_data


def sum_training(X, window_size=9, shuffle=True):
    """
    Performs summation on the training, so they can correspond to the neighbouring summation of the test set
    :param X: The X of the training set to perform summation on
    :param window_size: The number of samples to be summed
    :param shuffle: If true, shuffles the samples before summation, so it can be done uniformly
    :return: The X after summation, with its first dimension decreased by window_size
    """
    n, m = X.shape
    output = np.zeros((n - window_size + 1, m))

    if shuffle:
        shuffled_indices = np.arange(n)
        np.random.shuffle(shuffled_indices)
        X = X[shuffled_indices]

    for i in range(n - window_size + 1):
        output[i] = np.sum(X[i:i + window_size], axis=0)
    return output


def get_random_pixels(array_2D, n):
    """
    Picks random pixels from a 2D array
    :param array_2D: The 2D array that the pixels will be on
    :param n: The number of the selected pixels
    :return: The indices on the first two dimensions of the selected pixels
    """
    shape = array_2D.shape
    rand_indices_0 = np.random.randint(0, shape[0], size=n)
    rand_indices_1 = np.random.randint(0, shape[1], size=n)
    result = np.array([array_2D[rand_indices_0[i], rand_indices_1[i]] for i in range(n)])
    return result


classes = ['UNCL', 'PP', 'PE', 'PET', 'PVC']
Xs = construct_Xs(classes)
ys = construct_ys(Xs)

X_train = np.concatenate(Xs)
y_train = np.concatenate(ys)
np.save('X_train', X_train)
np.save('y_train', y_train)
