import json

import numpy as np
from matplotlib import pyplot as plt

from models.load_data import testing_indexes

for i in range(len(testing_indexes)):
    print("---------------------------", testing_indexes[i], "---------------------------")
    findPeaks = json.load(open('.\\findPeaks' + str(testing_indexes[i]) + '.json'))
    lstm = json.load(open('.\\97lstm' + str(testing_indexes[i]) + '.json'))
    windows = json.load(open('.\\1322windows' + str(testing_indexes[i]) + '.json'))
    conv = json.load(open('.\\299conv' + str(testing_indexes[i]) + '.json'))

    plt.figure(figsize=(10, 6))
    plt.plot(np.array(lstm['output']) * 0.2, label='LSTM', color='blue', linewidth=2)
    plt.plot(np.array(conv['output']) * 0.4, label='CNN', color='green', linewidth=2)
    plt.plot(np.array(windows['output']) * 0.6, label='Sliding Windows', color='purple', linewidth=2)
    plt.plot(np.array(findPeaks['output']) * 0.8, label='findPeaks', color='grey', linewidth=2)
    plt.plot(np.array(findPeaks['real']), label='Wartości prawdziwe', color='red', linewidth=2)
    plt.title('Porównanie modeli uczenia maszynowego i algorytmu findPeaks')
    plt.xlabel('Indeks')
    plt.ylabel('Wartość')
    plt.legend()
    plt.show()
