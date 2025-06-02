import json

from matplotlib import pyplot as plt

# d = json.load(open(".\\fixed-not-together-windows.json"))
#
# plt.plot(d['accuracy'], label='Accuracy', color='blue', linewidth=1)
# plt.plot(d['precision'], label='Precision', color='orange', linewidth=1)
# plt.plot(d['recall'], label='Recall', color='red', linewidth=1)
# plt.title("Wartości metryk w trakcie uczenia modelu")
# plt.ylabel('Wartość')
# plt.xlabel('Epoka')
# plt.legend()
# plt.show()

ep = " z $\\epsilon$ ("
hline = "\t\\hline"
models = ["conv", "lstm", "windows"]
configurations = ["not-fixed-not-together", "not-fixed-together", "fixed-not-together", "fixed-together"]

for model in models:
    print()
    print("---------------", model, "---------------")
    print()
    i = 5
    for config in configurations:
        d = json.load(open('.\\' + config + '-' + model + '.json'))
        best_index = 0
        best_value = 0
        for curr_index in range(len(d['recall'])):
            curr_value = (d['accuracy'][curr_index] + d['precision'][curr_index] + d['recall'][curr_index]) / 3
            if curr_value > best_value:
                best_value = curr_value
                best_index = curr_index
        print(hline)
        # Accuracy Precision Recall Średnia
        final_string = (config + ep + str(best_index) + ") & "
              + str("{:.3f}".format(d['accuracy'][best_index])) + " & "
              + str("{:.3f}".format(d['precision'][best_index])) + " & "
              + str("{:.3f}".format(d['recall'][best_index])) + " & " + "{:.3f}".format(best_value) + " \\\\")
        final_string = final_string.replace(".", ",")
        print("\t"+str(i) + '. & ' + final_string)
        i = i + 1
    print(hline)