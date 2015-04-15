#!/usr/bin/env python
# -*- encoding: utf-8 -*-

""" Optimal trajectory builder (Alex Bashuk's thesis work)."""

from copy import deepcopy
from bisect import bisect_left
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Alex Bashuk"
__copyright__ = "Copyright (c) 2015 Alex Bashuk"
__credits__ = ["Alex Bashuk"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Alex Bashuk"
__email__ = "alex@bashuk.tk"
__status__ = "Development"

class SplineBuilder:
    """Spline builder class.
    This class helps to build a cubic spline based on finite set of points,
    function values in these points, and boundary conditions of first type
    given on left and right sides of the interval."""

    def __init__(self):
        # Amount of points
        self._n = 0
        # Coordinates and function values
        self._x = []
        self._y = []
        # Cubic spline parameters
        self._a = []
        self._b = []
        self._c = []
        self._d = []

    def build(self, x, y, der_left = 0, der_right = 0):
        """Builds a spline."""

        if len(x) != len(y):
            raise ValueError("x and y have to be of the same size.")
        if len(x) < 2:
            raise ValueError("x must be at least of size 2.")

        self._n = len(x)
        self._x = deepcopy(x)
        self._y = deepcopy(y)

        # TODO: implement a, b, c, d computation

    def f(self, x):
        """Calculates the value of a spline approximation in a given point."""

        if self._n == 0:
            raise Exception("Spline not built yet.")
        if x < self._x[0] or self._x[-1] <= x:
            raise ValueError("Given point is out of interval.")

        index = bisect_left(self._x, x)
        x0 = self._x[index]
        a = self._a[index]
        b = self._b[index]
        c = self._c[index]
        d = self._d[index]
        value = a + b * (x - x0) + c * (x - x0) ** 2 + d * (x - x0) ** 3

        return value



if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 95)
    y = np.sin(x)

    subx = [x[i] for i in range(len(x)) if i % 10 == 0 and i > 0]
    sb = SplineBuilder()
    sb.build(x, y)
    z = [sb.f(point) for point in subx]


    plt.plot(x, y, subx, z)
    plt.show()

































