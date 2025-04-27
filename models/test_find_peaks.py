from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import numpy as np

from load_data import testing_indexes
from utils import prepare_model_data

is_fixed = True
is_together = True
checkpoint_to_test = 2
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'lstm'

testing_input_size = 2500
x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)

def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr

width = 30
order = 2
peakHeight = .20
prominence = .2
peakWitdh = 50

def findPeaks(serie_x):
    serie = normalize_array(serie_x)
    # remove noise applying savgol filter
    def suavizar(serie):
        from scipy.signal import savgol_filter
        suave = savgol_filter(serie, window_length=width, polyorder=order)
        return suave

    suave = suavizar(serie)
    # print(serie)
    peaks, _ = find_peaks(suave, height=peakHeight, prominence=prominence, distance=peakWitdh)
    # print(peaks)
    # print(len(peaks)) # number of found peaks

    found_peaks = np.zeros(testing_input_size)

    for p in peaks:
        found_peaks[p] = 1

    return found_peaks

for i in range(len(testing_indexes)):
    output = findPeaks(x[i])
    plt.figure(figsize=(10, 6))
    plt.plot(output, label='Output', color='blue', linewidth=1)
    plt.plot(y[i], label='Real values', color='red', linewidth=1)
    if is_together:
        plt.plot(x[i], label='Sensors', color='green', linewidth=1)
    else:
        interior, exterior = x[i][:, 0], x[i][:, 1]
        plt.plot(interior, label='Interior', color='orange', linewidth=1)
        plt.plot(exterior, label='Exterior', color='green', linewidth=1)
    plt.title('Porównanie wyjścia modelu i danych referencyjnych')
    plt.xlabel('Indeks')
    plt.ylabel('Wartość')
    plt.legend()
    plt.show()
