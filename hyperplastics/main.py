import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spectral import *
from sklearn.svm import SVC

from preprocessing import *
from plot import *
from utils import *
from peakann import *


if __name__ == '__main__':
    wl_img = open_image('../../RU/data/River_Plastics_26032024/PP_film_2024-03-26_07-34-06/'
                        'capture/PP_film_2024-03-26_07-34-06.hdr')
    wavelengths = wl_img.metadata['wavelength'][40:180]
    peakann_columns = ['Num of Peaks', 'Peak Int Sum', 'Peak Dist Sum', 'Rel Int']

    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    X_train = area_normalization(X_train)

    X_train = pd.DataFrame(data=X_train, index=range(1, X_train.shape[0] + 1), columns=wavelengths)
    y_train = pd.DataFrame(data=y_train, index=range(1, y_train.shape[0] + 1), columns=['Class'])

    clf = SVC(gamma='auto')
    clf.fit(X_train.values, y_train.values.ravel())
    print("SVC Fitting Completed!")

    # test_img = open_image('../../RU/data/6_1_NET3_2024-05-23_12-33-53/capture/6_1_NET3_2024-05-23_12-33-53.hdr')
    test_img = np.load('data/6_1/NET1_6_1.npy')
    test_img = test_img[::6, ::6, :]
    initial_shape = test_img.shape

    test_img = neighbouring_summation(test_img)

    X_test = reshape_3D_to_2D(test_img)
    X_test = savgol(X_test, window_length=5, polyorder=2, deriv=1)
    X_test = area_normalization(X_test)

    X_test = pd.DataFrame(data=X_test, index=range(1, X_test.shape[0] + 1), columns=wavelengths)

    y_pred = clf.predict(X_test.values)
    unique, counts = np.unique(y_pred, return_counts=True)
    print(dict(zip(unique, counts)))

    plot_predictions(y_pred, initial_shape, title='7x7 NS + SG + AreaNorm / SVC Predictions on 6/1 NET1')
