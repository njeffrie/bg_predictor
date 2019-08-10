"""
Calculate insulin on baord based on an exponential algorithm outlined here:
https://github.com/LoopKit/Loop/issues/388#issuecomment-317938473
"""
import numpy as np
import math

peak = 75.0
end = 300.0


def time_series_iob(insulin_values):
    iob_array = np.zeros(len(insulin_values))
    for i in range(len(insulin_values)):
        iob = 0
        for j in range(i):
            bolus_units = insulin_values[j]
            if (bolus_units > 0):
                mins_ago = (i - j) * 5
                activity_contrib, iob_contrib = calculate_iob_exponential(
                    bolus_units, mins_ago, end, peak)
                iob += iob_contrib
        iob_array[i] = iob
    return iob_array


def calculate_iob_exponential(insulin_units, mins_ago, duration, peak):
    if mins_ago < 0 or mins_ago >= duration:
        return 0.0, 0.0
    # Ensure we are working with floats, since non-float math will cause
    # implicit rounding.
    units = float(insulin_units)
    t = float(mins_ago)
    tp = float(peak)
    td = float(duration)
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    S = 1 / (1 - a + (1 + a) * math.exp(-td / tau))

    activity_contrib = units * ((S / math.pow(tau, 2)) * t *
                                        (1 - t / td) * math.exp(-t / tau))
    iob_contrib = units * (1 - S * (1 - a) * (
        (math.pow(t, 2) / (tau * td *
                           (1 - a)) - t / tau - 1) * math.exp(-t / tau) + 1))

    return activity_contrib, iob_contrib
