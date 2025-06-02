import numpy as np
import tensorflow as tf

from load_data import get_interior, get_true_output, get_exterior, get_fixed_output


def weighted_binary_crossentropy(y_true, y_pred):
    ones_count = tf.reduce_sum(y_true)
    zeros_count = tf.cast(tf.size(y_true, out_type=tf.int32), tf.float32) - ones_count
    total_count = tf.cast(tf.size(y_true, out_type=tf.int32), tf.float32)

    one_weight = zeros_count / total_count * 100
    zero_weight = ones_count / total_count * 100
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    bce = y_true * tf.math.log(y_pred + epsilon)
    bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
    bce = -bce
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_bce = weight_vector * bce
    return tf.reduce_mean(weighted_bce)


def prepare_model_data(data_indexes, input_size, is_fixed, is_together):
    x = []
    y = []
    for i in data_indexes:
        interior = get_interior(i)[-input_size:]
        exterior = get_exterior(i)[-input_size:]
        output_package = get_fixed_output(i)[-input_size:] if is_fixed else get_true_output(i)[-input_size:]
        new_x = np.array(np.array(interior) * np.array(exterior)) if is_together else np.column_stack((interior, exterior))
        new_y = np.array(output_package)
        x.append(new_x)
        y.append(new_y)
    return np.array(x), np.array(y)


def create_windows(x_org, y_org, input_size, accepting_peak_half_window, window_size, stride):
    x = []
    y = []
    for i in range(len(y_org)):
        new_x = x_org[i]
        new_y = y_org[i]
        for j in range((input_size - window_size) // stride + 1):
            x.append(np.array(new_x[j * stride: j * stride + window_size]))
            y_result = np.array(new_y[j * stride: j * stride + window_size])
            y.append(y_result[window_size // 2])
    return np.array(x), np.array(y)
