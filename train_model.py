import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import generate_sample_data as generator

print(tf.version.VERSION)
print(tf.keras.__version__)


def generate_model(result_len):
    model = tf.keras.Sequential()
    """
    model.add(layers.Dense(144, activation='relu', input_shape=(3, 72, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(36, activation='relu'))
    model.add(layers.Dense(result_len, activation='relu'))
    """
    model.add(
        layers.Convolution2D(32,
                             3,
                             3,
                             activation='relu',
                             input_shape=(3, 72, 1)))
    model.add(layers.AveragePooling2D(pool_size=(1, 2)))
    model.add(layers.Dense(144, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(1, 2)))
    model.add(layers.Dense(72, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(result_len, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model


if __name__ == "__main__":
    result_len = 1
    total_samples = 72 * 6
    inputs = np.zeros((3, total_samples, 1), dtype=float)
    generator.generate_model_values(inputs[0], inputs[1], inputs[2])
    model = generate_model(result_len)

    input_data = np.zeros((72 * 4, 3, 72, 1), dtype=float)
    labels = np.zeros((72 * 4, result_len), dtype=float)
    for i in range(72 * 4):
        input_data[i] = inputs[:, i:i + 72]
        labels[i] = inputs[0, i + 73:i + 73 + result_len, 0]
    print(input_data.shape)
    print(labels.shape)

    model.fit(input_data, labels, validation_split=0.2, epochs=300)

    test_input = np.zeros((1, 3, 72, 1), dtype=float)
    generator.generate_bg_values(test_input[0, 0], 0, 36, 100, 100)
    generator.eat_food(test_input[0, 1], 36, 100)
    generator.generate_bg_values(test_input[0, 0], 36, 36, 100, 135)
    print(model.predict(test_input))
