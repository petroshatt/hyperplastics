import numpy as np
import pandas as pd
from spectral import *

from sklearn.svm import SVC

from preprocessing import *
from plot import *
from utils import *


if __name__ == '__main__':
    wl_img = open_image('../../RU/data/River_Plastics_26032024/PP_film_2024-03-26_07-34-06/'
                        'capture/PP_film_2024-03-26_07-34-06.hdr')
    wavelengths = wl_img.metadata['wavelength'][40:180]

    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    X_train = savgol(X_train)
    X_train = area_normalization(X_train)

    X_train = pd.DataFrame(data=X_train, index=range(1, X_train.shape[0] + 1), columns=wavelengths)
    y_train = pd.DataFrame(data=y_train, index=range(1, y_train.shape[0] + 1), columns=['Class'])

    clf = SVC(gamma='auto')
    clf.fit(X_train.values, y_train.values.ravel())
    print("SVC Fitting Completed!")

    test_img = np.load('data/test_img.npy')
    test_img = test_img[::10, ::10, :]
    initial_shape = test_img.shape

    test_img = neighbouring_summation(test_img)

    X_test = reshape_3D_to_2D(test_img)

    X_test = savgol(X_test)
    X_test = area_normalization(X_test)
    X_test = pd.DataFrame(data=X_test, index=range(1, X_test.shape[0] + 1), columns=wavelengths)

    y_pred = clf.predict(X_test.values)
    unique, counts = np.unique(y_pred, return_counts=True)
    print(dict(zip(unique, counts)))

    plot_predictions(y_pred, initial_shape, title='5x5 NS + SG + AreaNorm / SVC Predictions')
