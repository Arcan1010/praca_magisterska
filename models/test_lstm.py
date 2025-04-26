import json

import matplotlib.pyplot as plt
import numpy as np
from keras.src.saving import load_model
from keras.src.utils import set_random_seed

from load_data import testing_indexes
from utils import prepare_model_data

set_random_seed(12345)

is_fixed = True
is_together = True
checkpoint_to_test = 2
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'lstm'
model = load_model('..\\checkpoints\\lstm\\' + identifier + '-' + str(checkpoint_to_test) + '.keras', compile=False)

model.summary()

testing_input_size = 2500
x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)

for i in range(len(testing_indexes)):
    output = model.predict(x[i].reshape(1, testing_input_size, 1 if is_together else 2)).reshape(testing_input_size)
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

# -------------------------------------------------------------------------------------------------------------------- #
#
#     output1 = model.predict(x[i][:2500].reshape(1, 2500, 1 if is_together else 2)).reshape(2500)
#     output2 = model.predict(x[i][2500:5000].reshape(1, 2500, 1 if is_together else 2)).reshape(2500)
#     output3 = model.predict(x[i][5000:7500].reshape(1, 2500, 1 if is_together else 2)).reshape(2500)
#     output4 = model.predict(x[i][7500:].reshape(1, 2500, 1 if is_together else 2)).reshape(2500)
#
#     output = np.concatenate((output1, output2, output3, output4), axis=0)
#
#     threshold = 0.5
#
#     def above_threshold(value):
#         return 1 if value >= threshold else 0
#
#
#     above_threshold_vectorized = np.vectorize(above_threshold)
#     output = above_threshold_vectorized(output)
#
#     my_dict = dict()
#     my_dict['output'] = output.tolist()
#     my_dict['real_values'] = y[i].tolist()
#     if is_together:
#         my_dict['sensors'] = x[i].tolist()
#     else:
#         interior, exterior = x[i][:, 0].tolist(), x[i][:, 1].tolist()
#         my_dict['interior'] = interior
#         my_dict['exterior'] = exterior
#
#     j = json.dumps(my_dict)
#     with open("..\\presentation\\" + identifier + '-' + str(checkpoint_to_test) + '.json', "w") as file:
#         json.dump(j, file)
# -------------------------------------------------------------------------------------------------------------------- #
