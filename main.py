#!/usr/bin/env python
# -*- encoding: utf-8 -*-

""" Optimal trajectory builder (Alex Bashuk's thesis work)."""

import copy
import bisect
import numpy as np
import matplotlib.pyplot as plt
import random

__author__ = "Alex Bashuk"
__copyright__ = "Copyright (c) 2015 Alex Bashuk"
__credits__ = "Alex Bashuk"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Alex Bashuk"
__email__ = "alex@bashuk.tk"
__status__ = "Development"

class SplineBuilder:
    """
    Spline builder class.
    This class helps to build a cubic spline based on finite set of points,
    function values in these points, and boundary conditions of first type
    given on left and right sides of the interval.
    """
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

    def _tdma_solve(self, a, b, c, d):
        """
        Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver

        TDMA solver, a b c d can be NumPy array type or Python list type.
        refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

        Source: https://gist.github.com/ofan666/1875903
        """
        # preappending missing parts
        a = [0] + a
        c = c + [0]

        nf = len(a)     # number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
        for it in xrange(1, nf):
            mc = ac[it]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
     
        xc = ac
        xc[-1] = dc[-1]/bc[-1]
     
        for il in xrange(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
     
        del bc, cc, dc  # delete variables from memory
     
        return list(xc)

    def build(self, x, y, der_left = 0, der_right = 0):
        """
        Builds a spline.
        Source: http://matlab.exponenta.ru/spline/book1/12.php
        """
        if len(x) != len(y):
            raise ValueError("x and y have to be of the same size.")
        if len(x) < 2:
            raise ValueError("x must be at least of size 2.")

        self._n = len(x) - 1
        self._x = copy.deepcopy(x)
        self._y = copy.deepcopy(y)

        h = []
        for k in range(self._n):
            h.append(x[k + 1] - x[k])

        if self._n > 1:
            # here must be (n - 1) equations for b
            alpha = []
            for k in range(1, self._n - 1):
                alpha.append(1.0 / h[k])
            beta = []
            for k in range(0, self._n - 1):
                beta.append(2.0 * (1.0 / h[k] + 1.0 / h[k + 1]))
            gamma = []
            for k in range(0, self._n - 2):
                gamma.append(1.0 / h[k + 1])
            delta = []
            for k in range(0, self._n - 1):
                delta.append(3.0 * (
                    (y[k + 2] - y[k + 1]) / (x[k + 2] - x[k + 1]) / h[k + 1] +
                    (y[k + 1] - y[k]) / (x[k + 1] - x[k]) / h[k]
                    ))
            # boundary conditions of 1st type
            delta[0] -= der_left * 1.0 / h[0]
            delta[self._n - 2] -= der_right * 1.0 / h[self._n - 1]
            self._b = self._tdma_solve(alpha, beta, gamma, delta)
        else:
            self._b = []
        
        self._a = copy.deepcopy(self._y)
        self._b = [der_left] + self._b + [der_right]
        self._c = []
        for k in range(0, self._n):
            self._c.append(
                (3.0 * (y[k + 1] - y[k]) / (x[k + 1] - x[k]) - 
                    self._b[k + 1] - 2.0 * self._b[k]) / 
                h[k]
                )
        self._d = []
        for k in range(0, self._n):
            self._d.append(
                (self._b[k] + self._b[k + 1] - 
                    2.0 * (y[k + 1] - y[k]) / (x[k + 1] - x[k])) / 
                (h[k] * h[k])
                )
        self._a = self._a[:-1]
        self._b = self._b[:-1]

        # TODO: implement a, b, c, d computation

    def f(self, x):
        """
        Calculates the value of a spline approximation in a given point.
        """
        if self._n == 0:
            raise Exception("Spline not built yet.")
        if x < self._x[0] or self._x[-1] < x:
            raise ValueError("Given point is out of interval.")

        if x == self._x[-1]:
            return self._y[-1]

        index = bisect.bisect_right(self._x, x) - 1
        x0 = self._x[index]
        a = self._a[index]
        b = self._b[index]
        c = self._c[index]
        d = self._d[index]
        value = a + b * (x - x0) + c * (x - x0) ** 2 + d * (x - x0) ** 3

        return value

class QualityFunctionBuilder:
    """
    Quality function builder class.
    This class helps to load the terrain quality function Q(x, y) from file.
    Also it provides methods for calculating the values of points between
    the pixel centers.
    """
    def __init__(self):
        # image weight and height
        self._imw = 0
        self._imh = 0
        # terrain pixel data
        self._im = []
        # terrain weight and height
        self.w = 0
        self.h = 0

    def load_from_image(self, filename):
        """
        Load quality function from image file.
        Pixels with high values (white) correspond to bigger height on the
        terrain.
        This method sets terrain size to be equal to the image size.
        To change the terrain size, use set_custom_terrain_size() method.
        """
        pass

    def set_custom_terrain_size(self, w, h):
        """
        Sets custom terrain size (terrain sizes equals image size by default).
        """
        if w <= 0 or h <= 0:
            raise ValueError("Terrain size must be positive.")
        self.w = w
        self.h = h

    def Q(self, x, y):
        """
        Quality function, linearly interpolated from the image pixel values.
        """
        if self._w == 0:
            raise Exception("Quality funcion not loaded yet.")
        if x < 0 or x >= self.w or y < 0 or y >= self.y:
            raise ValueError("Given point is out of the terrain.")

class Tester:
    """
    Tester class.
    Contains methods for testing purposes.
    """
    def test_spline_builder(self, sample_size = 5):
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)

        sub_x = [x[0]] + sorted(random.sample(x[1 : -1], sample_size)) + [x[-1]]
        sub_y = np.sin(sub_x)

        sb = SplineBuilder()
        sb.build(sub_x, sub_y, -1.0, -1.0)
        z = [sb.f(point) for point in x]

        plt.plot(x, y, 'b--', sub_x, sub_y, 'ro', x, z, 'g')
        plt.show()

if __name__ == '__main__':
    # Tester().test_spline_builder()
    pass






























