import json

import keras
import numpy as np
from keras import Sequential
from keras.src.layers import Dense
from keras.src.metrics import Precision, Recall
from keras.src.utils import set_random_seed
from matplotlib import pyplot as plt

from load_data import training_indexes
from utils import prepare_model_data, create_windows, weighted_binary_crossentropy

set_random_seed(12345)

is_fixed = True
is_together = False
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'windows'
checkpoint_path = '..\\checkpoints\\windows\\' + identifier + '-{epoch:1d}.keras'
window_size = 250
stride = 50
accepting_peak_half_window = 5
training_input_size = 10_000
epochs = 2000

print(identifier)

x_train, y_train = prepare_model_data(training_indexes, training_input_size, is_fixed, is_together)
x_train_np, y_train_np = create_windows(x_train, y_train, training_input_size, accepting_peak_half_window, window_size, stride)

x_train_np_temp = []
for i in range(x_train_np.shape[0]):
    x_train_np_temp.append(x_train_np[i].flatten())
x_train_np = np.array(x_train_np_temp)

model = Sequential([
    Dense(window_size * (1 if is_together else 2), activation='relu'),
    Dense(100, activation='relu'),
    Dense(50),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy', Precision(), Recall()])
model.summary()

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

history = model.fit(np.array(x_train_np), np.array(y_train_np), epochs=epochs,
          batch_size=len(training_indexes), callbacks=[model_checkpoint_callback])

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
