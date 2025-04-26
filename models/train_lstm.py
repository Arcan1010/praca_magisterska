import json

import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.src.layers import LSTM, InputLayer
from keras.src.metrics import Precision, Recall
from keras.src.utils import set_random_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from load_data import training_indexes
from utils import weighted_binary_crossentropy, prepare_model_data, weighted_binary_crossentropy_with_static_weights

set_random_seed(12345)

is_fixed = True
is_together = True
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'lstm'
checkpoint_path = '..\\checkpoints\\lstm\\' + identifier + '-{epoch:1d}.keras'
build_training_set_from_size = 10_000
training_input_size = 2500
epochs = 100

print(identifier)

x_train_np, y_train_np = prepare_model_data(training_indexes, build_training_set_from_size, is_fixed, is_together)

whole_x = []
whole_y = []
for i in range(x_train_np.shape[0]):
    temp_x = []
    temp_y = []
    divider = build_training_set_from_size // training_input_size
    for j in range(divider):
        hx = x_train_np[i][j * training_input_size: (j + 1) * training_input_size]
        hy = y_train_np[i][j * training_input_size: (j + 1) * training_input_size]
        temp_x.append(np.array(hx))
        temp_y.append(np.array(hy))
    whole_x = whole_x + temp_x
    whole_y = whole_y + temp_y

x_train_np = np.array(whole_x)
y_train_np = np.array(whole_y)

hidden_layer_size = 250
model = Sequential([
    InputLayer(shape=(training_input_size, 1 if is_together else 2)),
    LSTM(hidden_layer_size, return_sequences=True),
    LSTM(hidden_layer_size, return_sequences=True),
    LSTM(hidden_layer_size, return_sequences=True),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy', Precision(), Recall()])
model.summary()

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

history = model.fit(x_train_np, y_train_np, epochs=epochs, batch_size=len(training_indexes), callbacks=[model_checkpoint_callback])

with open('..\\' + identifier + '.json', "w") as text_file:
    text_file.write(json.dumps(history.history))

plt.plot(history.history['accuracy'], label='Accuracy', color='blue', linewidth=1)
plt.plot(history.history['precision'], label='Precision', color='orange', linewidth=1)
plt.plot(history.history['recall'], label='Recall', color='red', linewidth=1)
plt.title(identifier)
plt.ylabel('Wartość')
plt.xlabel('Epoka')
plt.legend()
plt.savefig('..\\' + identifier + '.png')
