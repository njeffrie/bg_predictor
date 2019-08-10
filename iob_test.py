import unittest

import iob
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []

"""
Test suite for iob calculator
"""
class TestIobCalculator(unittest.TestCase):
    def test_out_of_range(self):
        act, iob_contrib = iob.calculate_iob_exponential(10, 600, 300, 60)
        self.assertLess(act, 0.01)
        self.assertLess(iob_contrib, 0.01)

    def test_time_zero(self):
        act, iob_contrib = iob.calculate_iob_exponential(10, 0, 300, 60)
        self.assertEqual(act, 0)
        self.assertEqual(iob_contrib, 10.0)

    def test_plot(self):
        for i in range(700):
            act, iob_contrib = iob.calculate_iob_exponential(1.0, float(i), 300.0, 75.0)
            xdata.append(i)
            ydata.append(act)
        ax.set_xlim(0, 700)
        ax.set_ylim(0, .01)
        line.set_data(xdata, ydata)
        plt.show()

if __name__ == '__main__':
    unittest.main()
