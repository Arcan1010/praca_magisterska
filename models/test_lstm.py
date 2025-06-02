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
checkpoint_to_test = 97
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'lstm'
model = load_model('..\\checkpoints\\lstm\\' + identifier + '-' + str(checkpoint_to_test) + '.keras', compile=False)

model.summary()

testing_input_size = 10_000
x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)



# configurations = ["not-fixed-not-together", "not-fixed-together", "fixed-not-together", "fixed-together"]
#
# ep = " bez $\\epsilon$ ("
# hline = "\t\\hline"
#
# for config in configurations:
#     d = json.load(open('..\\learned\\' + config + '-' + 'lstm' + '.json'))
#     best_index = 0
#     best_value = 0
#     for curr_index in range(len(d['recall'])):
#         curr_value = (d['accuracy'][curr_index] + d['precision'][curr_index] + d['recall'][curr_index]) / 3
#         if curr_value > best_value:
#             best_value = curr_value
#             best_index = curr_index
#     # Accuracy Precision Recall
#     model = load_model('..\\checkpoints\\' + 'lstm' + '\\' + config + '-lstm-' + str(best_index) + '.keras', compile=False)
#     accuracy_sum = 0
#     precision_sum = 0
#     recall_sum = 0
#
#     is_together = "not-together" not in config
#     is_fixed = "not-fixed" not in config
#
#     testing_input_size = 2500
#     x, y = prepare_model_data(testing_indexes, testing_input_size, is_fixed, is_together)
#
#     for i in range(len(testing_indexes)):
#         output = model.predict(x[i].reshape(1, testing_input_size, 1 if is_together else 2), verbose = None).reshape(testing_input_size)
#         threshold = 0.5
#
#         def above_threshold(value):
#             return 1 if value >= threshold else 0
#
#         above_threshold_vectorized = np.vectorize(above_threshold)
#         output = above_threshold_vectorized(output)
#
#         accuracy = 0
#         for value_index in range(len(output)):
#             accuracy = accuracy + (1 if output[value_index] == y[i][value_index] else 0)
#         accuracy = accuracy / len(output)
#
#         precision = 0
#         precision_denominator = 0
#         for value_index in range(len(output)):
#             precision = precision + (
#                 1 if (output[value_index] == y[i][value_index] and output[value_index] == 1) else 0)
#             precision_denominator = precision_denominator + (1 if output[value_index] == 1 else 0)
#         precision = precision / precision_denominator
#
#         recall = 0
#         recall_denominator = 0
#         for value_index in range(len(output)):
#             recall = recall + (1 if (output[value_index] == y[i][value_index] and output[value_index] == 1) else 0)
#             recall_denominator = recall_denominator + (1 if y[i][value_index] == 1 else 0)
#         recall = (recall / recall_denominator) if recall_denominator != 0 else 1
#
#         accuracy_sum = accuracy_sum + accuracy
#         precision_sum = precision_sum + precision
#         recall_sum = recall_sum + recall
#     accuracy_sum = accuracy_sum / len(testing_indexes)
#     precision_sum = precision_sum / len(testing_indexes)
#     recall_sum = recall_sum / len(testing_indexes)
#
#     print(hline)
#     final_string = (config + ep + str(best_index) + ") & "
#                     + str("{:.3f}".format(accuracy_sum)) + " & "
#                     + str("{:.3f}".format(precision_sum)) + " & "
#                     + str("{:.3f}".format(recall_sum)) + "\\\\")
#     final_string = final_string.replace(".", ",")
#     print("\t" + str(0) + '. & ' + final_string)



for i in range(len(testing_indexes)):
    output = model.predict(x[i].reshape(1, testing_input_size, 1 if is_together else 2)).reshape(testing_input_size)

    threshold = 0.5
    def above_threshold(value):
        return 1 if value >= threshold else 0

    above_threshold_vectorized = np.vectorize(above_threshold)
    output = above_threshold_vectorized(output)

    # Zapis wynikow do pliku #
    import json

    dane = {
        'output': np.array(output).tolist(),
        'real': y[i].tolist(),
        'sensors': x[i].tolist()
    }

    print(len(y[i]))

    # with open('..\\pliki-do-podsumowania\\' + str(checkpoint_to_test) + 'lstm' + str(testing_indexes[i]) + '.json',
    #           'w') as plik:
    #     json.dump(dane, plik, indent=4)

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
    # plt.xlim(5500, 6700)
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
