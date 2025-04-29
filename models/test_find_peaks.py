from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import numpy as np

from load_data import testing_indexes
from utils import prepare_model_data

is_fixed = True
is_together = True
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'lstm'

testing_input_size = 10_000
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

    print('Find peaks, is_fixed:', is_fixed, "testing index:", i)

    accuracy = 0
    for value_index in range(len(output)):
        accuracy = accuracy + (1 if output[value_index] == y[i][value_index] else 0)
    accuracy = accuracy / len(output)

    precision = 0
    precision_denominator = 0
    for value_index in range(len(output)):
        precision = precision + (1 if (output[value_index] == y[i][value_index] and output[value_index] == 1) else 0)
        precision_denominator = precision_denominator + (1 if output[value_index] == 1 else 0)
    precision = precision / precision_denominator

    recall = 0
    recall_denominator = 0
    for value_index in range(len(output)):
        recall = recall + (1 if (output[value_index] == y[i][value_index] and output[value_index] == 1) else 0)
        recall_denominator = recall_denominator + (1 if y[i][value_index] == 1 else 0)
    recall = recall / recall_denominator

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)

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
