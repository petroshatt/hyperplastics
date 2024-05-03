import numpy as np
import pandas as pd
from spectral import *

from sklearn.svm import SVC

from preprocessing import *


if __name__ == '__main__':
    wl_img = open_image('../../RU/data/River_Plastics_26032024/PP_film_2024-03-26_07-34-06/'
                        'capture/PP_film_2024-03-26_07-34-06.hdr')
    wavelengths = wl_img.metadata['wavelength'][40:180]

    X_train = np.load('training_set/X_train.npy')
    y_train = np.load('training_set/y_train.npy')
    X_train = pd.DataFrame(data=X_train, index=range(1, X_train.shape[0] + 1), columns=wavelengths)
    y_train = pd.DataFrame(data=y_train, index=range(1, y_train.shape[0] + 1), columns=['Class'])

    X_train = savgol(X_train)
    X_train = area_normalization(X_train)

    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train.values.ravel())
    print("SVC Fitting Completed!")
