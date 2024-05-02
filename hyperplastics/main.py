import numpy as np
import pandas as pd
from spectral import *

from sklearn.svm import SVC

from preprocessing import *


if __name__ == '__main__':
    x = open_image('../../../RU/data/quant_good_2024-04-10_11-43-07/capture/quant_good_2024-04-10_11-43-07.hdr')
    x = x[:, :, :]

    x2 = neighbouring_summation(x)
