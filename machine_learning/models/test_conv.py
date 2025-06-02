import matplotlib.pyplot as plt
import numpy as np
from keras.src.saving import load_model
from keras.src.utils import set_random_seed

from load_data import testing_indexes
from models.utils import prepare_model_data

set_random_seed(12345)

is_fixed = True
is_together = True
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'conv'
checkpoint_to_test = 299
model = load_model('..\\checkpoints\\conv\\' + identifier + '-' + str(checkpoint_to_test) + '.keras', compile=False)

model.summary()

testing_input_size = 10_000
window_size = 512
stride = window_size
x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)

for i in range(len(testing_indexes)):
    new_x_test = x[i]
    new_y_test = y[i]
    model_out = []

    for j in range((testing_input_size - window_size) // stride + 1):
        windowed_x = np.array(new_x_test[j * stride: j * stride + window_size])
        predicted = model.predict(windowed_x.reshape(1, window_size, 1 if is_together else 2))[0]
        model_out = model_out + predicted.tolist()

    threshold = 0.5
    def above_threshold(value):
        return 1 if value >= threshold else 0

    above_threshold_vectorized = np.vectorize(above_threshold)
    model_out = above_threshold_vectorized(model_out)

    plt.figure(figsize=(10, 6))
    plt.plot(np.array(model_out), label='Wyjście sieci', color='blue', linewidth=1)
    plt.plot(new_y_test, label='Wartości prawdziwe', color='red', linewidth=1)
    if is_together:
        plt.plot(x[i], label='Czujniki', color='green', linewidth=1)
    else:
        interior, exterior = x[i][:, 0], x[i][:, 1]
        plt.plot(interior, label='Interior', color=(1, 0.5, 0.0, 0.5), linewidth=1)
        plt.plot(exterior, label='Exterior', color=(0.0, 1.0, 0.0, 0.5), linewidth=1)
    plt.title('Porównanie wyjścia modelu i danych referencyjnych')
    plt.xlabel('Indeks')
    plt.ylabel('Wartość')
    plt.legend()
    plt.show()