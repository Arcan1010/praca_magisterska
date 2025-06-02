import json

import keras
import numpy as np
from keras import Sequential, Input, Model
from keras.src.layers import Conv1D, LeakyReLU, Conv1DTranspose, BatchNormalization, Dropout
from keras.src.metrics import Precision, Recall
from keras.src.optimizers import Adam
from keras.src.utils import set_random_seed
from matplotlib import pyplot as plt

from utils import weighted_binary_crossentropy, prepare_model_data, weighted_binary_crossentropy_with_static_weights
from load_data import training_indexes

set_random_seed(12345)

is_fixed = True
is_together = False
identifier = ('fixed-' if is_fixed else 'not-fixed-') + ('together-' if is_together else 'not-together-') + 'conv'
checkpoint_path = '..\\checkpoints\\conv\\' + identifier + '-{epoch:1d}.keras'
window_size = 512
stride = 50
training_input_size = 10_000
accepting_peak_half_window = 5

print(identifier)

epochs = 500

x, y = prepare_model_data(training_indexes, training_input_size, is_fixed, is_together)
x_train = []
y_train = []

for i in range(len(training_indexes)):
    new_x_train = x[i]
    new_y_train = y[i]
    for j in range((training_input_size - window_size) // stride + 1):
        x_train.append(np.array(new_x_train[j * stride: j * stride + window_size]))
        y_train.append(np.array(new_y_train[j * stride: j * stride + window_size]))

x_train_np = np.array(x_train)
y_train_np = np.array(y_train)

# -------------------------------------------------------------------------------------------------------------------- #

channel = 1 if is_together else 2
inputs = Input(shape=[window_size, channel])

def downsample(filters, size):
    result = Sequential()
    result.add(Conv1D(filters, size, strides=2, padding='same'))
    result.add(LeakyReLU(0.25))
    return result

def upsample(filters, size):
    result = Sequential()
    result.add(Conv1DTranspose(filters, size, strides=2 , padding='same'))
    result.add(BatchNormalization())
    result.add(Dropout(0.25))
    result.add(LeakyReLU(0.25))
    return result

down_stack = [
    downsample(16, 9), # (bs, 512, 1)
    downsample(16, 9), # (bs, 256, 1)
    downsample(32, 6), # (bs, 128, 1)
    downsample(32, 6), # (bs, 64, 1)
    downsample(64, 3), # (bs, 32, 1)
    downsample(64, 3), # (bs, 16, 1)
    ]

up_stack = [
    upsample(64, 3), # (bs, 16, 1)
    upsample(32, 3), # (bs, 32, 1)
    upsample(32, 6), # (bs, 64, 1)
    upsample(16, 6), # (bs, 128, 1)
    upsample(16, 9), # (bs, 256, 1)
    ]

last = Conv1DTranspose(1, 9, strides=2, padding='same', activation='sigmoid')  # (bs, 512, 1)

x = inputs

for down in down_stack:
    x = down(x)

for up in up_stack:
    x = up(x)

x = last(x)

model = Model(inputs=inputs, outputs=x)

# -------------------------------------------------------------------------------------------------------------------- #

optimizer = Adam(0.01)

model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy', Precision(), Recall()])
model.summary()

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

history = model.fit(np.array(x_train_np), np.array(y_train_np), epochs=epochs, batch_size=training_input_size, callbacks=[model_checkpoint_callback])

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
