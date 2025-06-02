import json

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
# checkpoint_to_test = 100
# model = load_model('..\\checkpoints\\conv\\' + identifier + '-' + str(checkpoint_to_test) + '.keras', compile=False)

# model.summary()

testing_input_size = 10_000
window_size = 512
stride = window_size
# x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)












configurations = ["not-fixed-not-together", "not-fixed-together", "fixed-not-together", "fixed-together"]
ep = " bez $\\epsilon$ ("
hline = "\t\\hline"

for config in configurations:
    d = json.load(open('..\\' + config + '-' + 'conv' + '.json'))
    best_index = 0
    best_value = 0
    for curr_index in range(len(d['recall'])):
        curr_value = (d['accuracy'][curr_index] + d['precision'][curr_index] + d['recall'][curr_index]) / 3
        if curr_value > best_value:
            best_value = curr_value
            best_index = curr_index
    # Accuracy Precision Recall
    model = load_model('..\\checkpoints\\' + 'conv' + '\\' + config + '-conv-' + str(best_index) + '.keras', compile=False)
    accuracy_sum = 0
    precision_sum = 0
    recall_sum = 0

    is_together = "not-together" not in config
    is_fixed = "not-fixed" not in config

    x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)
    for i in range(len(testing_indexes)):
        new_x_test = x[i]
        new_y_test = y[i]
        model_out = []

        for j in range((testing_input_size - window_size) // stride + 1):
            windowed_x = np.array(new_x_test[j * stride: j * stride + window_size])
            predicted = model.predict(windowed_x.reshape(1, window_size, 1 if is_together else 2), verbose = None)[0]
            model_out = model_out + predicted.tolist()

        threshold = 0.5
        def above_threshold(value):
            return 1 if value >= threshold else 0

        above_threshold_vectorized = np.vectorize(above_threshold)
        model_out = above_threshold_vectorized(model_out)

        output = np.array(model_out)
        y[i] = new_y_test

        accuracy = 0
        for value_index in range(len(output)):
            accuracy = accuracy + (1 if output[value_index] == y[i][value_index] else 0)
        accuracy = accuracy / len(output)

        precision = 0
        precision_denominator = 0
        for value_index in range(len(output)):
            precision = precision + (
                1 if (output[value_index] == y[i][value_index] and output[value_index] == 1) else 0)
            precision_denominator = precision_denominator + (1 if output[value_index] == 1 else 0)
        precision = precision / precision_denominator

        recall = 0
        recall_denominator = 0
        for value_index in range(len(output)):
            recall = recall + (1 if (output[value_index] == y[i][value_index] and output[value_index] == 1) else 0)
            recall_denominator = recall_denominator + (1 if y[i][value_index] == 1 else 0)
        recall = (recall / recall_denominator) if recall_denominator != 0 else 1

        accuracy_sum = accuracy_sum + accuracy
        precision_sum = precision_sum + precision
        recall_sum = recall_sum + recall
    accuracy_sum = accuracy_sum / len(testing_indexes)
    precision_sum = precision_sum / len(testing_indexes)
    recall_sum = recall_sum / len(testing_indexes)

    print(hline)
    final_string = (config + ep + str(best_index) + ") & "
                    + str("{:.3f}".format(accuracy_sum)) + " & "
                    + str("{:.3f}".format(precision_sum)) + " & "
                    + str("{:.3f}".format(recall_sum)) + "\\\\")
    final_string = final_string.replace(".", ",")
    print("\t" + str(0) + '. & ' + final_string)









# for i in range(len(testing_indexes)):
#     new_x_test = x[i]
#     new_y_test = y[i]
#     model_out = []
#
#     for j in range((testing_input_size - window_size) // stride + 1):
#         windowed_x = np.array(new_x_test[j * stride: j * stride + window_size])
#         predicted = model.predict(windowed_x.reshape(1, window_size, 1 if is_together else 2))[0]
#         model_out = model_out + predicted.tolist()
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.array(model_out), label='Output', color='blue', linewidth=1)
#     plt.plot(new_y_test, label='Real values', color='red', linewidth=1)
#     if is_together:
#         plt.plot(x[i], label='Sensors', color='green', linewidth=1)
#     else:
#         interior, exterior = x[i][:, 0], x[i][:, 1]
#         plt.plot(interior, label='Interior', color='orange', linewidth=1)
#         plt.plot(exterior, label='Exterior', color='green', linewidth=1)
#     plt.title('Porównanie wyjścia modelu i danych referencyjnych')
#     plt.xlabel('Indeks')
#     plt.ylabel('Wartość')
#     plt.legend()
#     plt.show()
#
#     threshold = 0.5
#     def above_threshold(value):
#         return 1 if value >= threshold else 0
#
#     above_threshold_vectorized = np.vectorize(above_threshold)
#     model_out = above_threshold_vectorized(model_out)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(np.array(model_out), label='Output', color='blue', linewidth=1)
#     plt.plot(new_y_test, label='Real values', color='red', linewidth=1)
#     if is_together:
#         plt.plot(x[i], label='Sensors', color='green', linewidth=1)
#     else:
#         interior, exterior = x[i][:, 0], x[i][:, 1]
#         plt.plot(interior, label='Interior', color=(1, 0.5, 0.0, 0.5), linewidth=1)
#         plt.plot(exterior, label='Exterior', color=(0.0, 1.0, 0.0, 0.5), linewidth=1)
#     plt.title('Porównanie wyjścia modelu i danych referencyjnych')
#     plt.xlabel('Indeks')
#     plt.ylabel('Wartość')
#     plt.legend()
#     plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
#     threshold = 0.5
#     def above_threshold(value):
#         return 1 if value >= threshold else 0
#
#     above_threshold_vectorized = np.vectorize(above_threshold)
#     model_out = above_threshold_vectorized(model_out)
#
#     my_dict = dict()
#     my_dict['output'] = np.array(model_out).tolist()
#     my_dict['real_values'] = new_y_test.tolist()
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