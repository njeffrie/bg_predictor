"""
Generate a time series of blood sugar values.  Simulate eating carbs and
forgetting to bolus, causing a temporary rise in blood sugar.  Not real or
accurate, just useful for training a sample model.
"""

import numpy as np
from random import randrange


def eat_food(food_values, idx, amount):
    food_values[idx] = amount


def take_insulin(insulin_values, idx, amount):
    insulin_values[idx] = amount


def generate_bg_values(bg_values, start_idx, num_values, start_value,
                       end_value):
    for i in range(num_values):
        value_approx = start_value + (end_value - start_value) * i / num_values
        bg_values[start_idx + i] = value_approx + randrange(-5, 5)


def generate_model_values(bg_values, food_values, insulin_values):
    chunk_size = len(bg_values) / 4

    # Eat food, followed by rising bg.
    eat_food(food_values, 0, 20)
    generate_bg_values(bg_values, 0, chunk_size, 100, 200)

    # high
    generate_bg_values(bg_values, chunk_size, chunk_size, 200, 200)

    # falling
    take_insulin(insulin_values, chunk_size, 3.5)
    generate_bg_values(bg_values, chunk_size * 2, chunk_size, 200, 100)

    # re-stabilized
    generate_bg_values(bg_values, chunk_size * 3, chunk_size, 100, 100)


if __name__ == "__main__":
    total_samples = 72
    inputs = np.zeros((3, total_samples), dtype=float)
    generate_model_values(inputs[0], inputs[1], inputs[2])
    print(inputs)
