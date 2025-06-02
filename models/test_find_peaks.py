from scipy.signal import find_peaks
from scipy.signal import savgol_filter

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

def findPeaks(series_x):
    normalized_series = normalize_array(series_x)
    smoothened_series = savgol_filter(normalized_series, window_length=30, polyorder=2)
    peaks, _ = find_peaks(smoothened_series, height=.20, prominence=.20, distance=50)
    found_peaks = np.zeros(testing_input_size)
    for p in peaks:
        found_peaks[p] = 1
    return found_peaks

accuracy_sum = 0
precision_sum = 0
recall_sum = 0

for i in range(len(testing_indexes)):
    output = findPeaks(x[i])

    #print('Find peaks, is_fixed:', is_fixed, "testing index:", testing_indexes[i])

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

    #print('Accuracy:', accuracy, 'Precision:', precision, 'Recall:', recall)
    #print('\hline')
    #print(accuracy, ' & ', precision, ' & ', recall, '\\\\')
    accuracy_sum = accuracy_sum + accuracy
    precision_sum = precision_sum + precision
    recall_sum = recall_sum + recall

    # Zapis wynikow do pliku #
    import json

    dane = {
        'output': output.tolist(),
        'real': y[i].tolist(),
        'sensors': x[i].tolist()
    }

    with open('..\\pliki-do-podsumowania\\findPeaks' + str(testing_indexes[i]) + '.json', 'w') as plik:
        json.dump(dane, plik, indent=4)


    plt.figure(figsize=(10, 6))
    plt.plot(output, label='Wyjście sieci', color='blue', linewidth=1)
    plt.plot(y[i], label='Wartości prawdziwe', color='red', linewidth=1)
    if is_together:
        plt.plot(x[i], label='Czujniki', color='green', linewidth=1)
    else:
        interior, exterior = x[i][:, 0], x[i][:, 1]
        plt.plot(interior, label='Interior', color='orange', linewidth=1)
        plt.plot(exterior, label='Exterior', color='green', linewidth=1)
    plt.title('Porównanie wyjścia modelu i danych referencyjnych')
    plt.xlabel('Indeks')
    plt.ylabel('Wartość')
    plt.legend()

    plt.xlim(5500, 6700)

    #plt.show()


# print(accuracy_sum/7, precision_sum/7, recall_sum/7)