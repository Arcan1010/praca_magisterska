import matplotlib.pyplot as plt
import numpy as np
from keras.src.saving import load_model
from keras.src.utils import set_random_seed

from load_data import testing_indexes
from utils import prepare_model_data

set_random_seed(12345)

is_fixed = True
is_together = True
checkpoint_to_test = 97
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'lstm'
model = load_model('..\\checkpoints\\lstm\\' + identifier + '-' + str(checkpoint_to_test) + '.keras', compile=False)

model.summary()

testing_input_size = 10_000
x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)

for i in range(len(testing_indexes)):
    output = model.predict(x[i].reshape(1, testing_input_size, 1 if is_together else 2)).reshape(testing_input_size)

    threshold = 0.5
    def above_threshold(value):
        return 1 if value >= threshold else 0

    above_threshold_vectorized = np.vectorize(above_threshold)
    output = above_threshold_vectorized(output)

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
