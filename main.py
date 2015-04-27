#!/usr/bin/env python
# -*- encoding: utf-8 -*-

""" Optimal trajectory builder (Alex Bashuk's thesis work)."""

import copy
import bisect
import numpy as np
import matplotlib.pyplot as plt
import random
import PIL
import os
import sys

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

        x = map(float, x)
        y = map(float, y)

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

    def f(self, x):
        """
        Calculates the value of a spline approximation in a given point.
        """
        if self._n == 0:
            raise Exception("Spline not built yet.")
        if x < self._x[0] or self._x[-1] < x:
            raise ValueError("Given point is out of interval.")

        x = float(x)

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
        # image data
        self._im = []
        # terrain weight and height
        self.w = 0.0
        self.h = 0.0

    def _build_plane_by_3_points(self, p1, p2, p3):
        """
        Builds a plane that goes through three given points in 3-D space.
        Plane is defined by equation: Ax + By + Cz + D = 0.
        Coefficients A, B, C, D are returned as a result.
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        v1 = p1 - p3
        v2 = p2 - p3
        norm = np.cross(v1, v2)

        A, B, C = map(float, norm)
        D = float(- np.dot(norm, p3))

        return A, B, C, D

    def load_from_image(self, filename):
        """
        Load quality function from image file.
        Pixels with high values (white) correspond to bigger height on the
        terrain.
        This method sets terrain size to be equal to the image size.
        To change the terrain size, use set_custom_terrain_size() method.
        """
        im = PIL.Image.open(filename)
        if im.mode != "L":
            im = im.convert("L")
        
        pix = im.load()
        self._imw, self._imh = im.size
        self.w, self.h = float(self._imw - 1), float(self._imh - 1)
        if self.w < 1 or self.h < 1:
            self.__init__()
            raise Exception("Image size should be at least 2x2.")
        self._im = [
            [float(pix[x, y]) for y in xrange(self._imh)] 
            for x in xrange(self._imw)
        ]

    def set_custom_terrain_size(self, size):
        """
        Sets custom terrain size (terrain sizes equal image size by default).
        """
        if self.w == 0:
            raise Exception("You should load an image before setting size.")

        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError("Terrain size must be positive.")
        self.w = float(w)
        self.h = float(h)

    def Q(self, x, y):
        """
        Quality function, linearly interpolated from the image pixel values.
        """
        if self.w == 0:
            raise Exception("Quality funcion not loaded yet.")

        if x < 0 or x > self.w or y < 0 or y > self.h:
            # I thought of the exception here at first, but then I realised
            # that splines can go beyond the road, and when calculating
            # the quality for that spline – we don't want it to raise any
            # exceptions, we just want it to make an enormously big jump
            return - 10 ** 9
            # raise ValueError("Given point is out of the terrain.")

        # Scaling
        x = 1.0 * x * (self._imw - 1) / self.w
        y = 1.0 * y  * (self._imh - 1) / self.h

        # Handling boundaries
        eps = 10 ** -5
        if abs(x) < eps and x <= 0.0:
            x += eps
        if abs(x - (self._imw - 1)) < eps:
            x -= eps
        if abs(y) < eps and y <= 0.0:
            y += eps
        if abs(y - (self._imh - 1)) < eps:
            y -= eps

        # From now on, coordinates here must be strictly inside the image:
        # 0.0 < x < _imw, 0.0 < y < _imh
        p1 = (int(x) + 1, int(y))
        p2 = (int(x), int(y) + 1)
        if x - int(x) + y - int(y) <= 1.0:
            p3 = (int(x), int(y))
        else:
            p3 = (int(x) + 1, int(y) + 1)
        
        p1 = (p1[0], p1[1], self._im[p1[0]][p1[1]])
        p2 = (p2[0], p2[1], self._im[p2[0]][p2[1]])
        p3 = (p3[0], p3[1], self._im[p3[0]][p3[1]])
        a, b, c, d = self._build_plane_by_3_points(p1, p2, p3)

        res = - (a * x + b * y + d) / c
        return res

class CarBuilder:
    """
    Simple class for containing car parameters.
    """
    def __init__(self, width = 17.1, length = 27.1, wheel = 3.15):
        """
        Default parameters (width = 17.1, length = 27.1, wheel = 3.15) are
        the actual parameters of Bugatti Veyron 16.4, scale 1:10.
        """
        self.width = width
        self.length = length
        self.wheel = wheel

class VehicleTrajectoryBuilder:
    """
    Vehicle Trajectory Builder.
    This class builds the trajectory for a 4-wheels vehicle. Vehicle's movement
    is defined by Ackermann steering geometry.
    """
    def __init__(self, qfb, car, fl = None, fr = None, dfl = None, dfr = None):
        if type(qfb) != QualityFunctionBuilder:
            raise TypeError("First argument must be a QFB.")

        if type(qfb) != QualityFunctionBuilder:
            raise TypeError("Second argument must be a car.")

        # Helper classes
        self._qfb = qfb
        self._sb = SplineBuilder()
        # Boundaries
        self._fl = fl if fl is not None else qfb.h * 0.5
        self._fr = fr if fr is not None else qfb.h * 0.5
        self._dfl = dfl if dfl is not None else 0.0
        self._dfr = dfr if dfr is not None else 0.0

    def _generate_straight_trajectory(self, points):
        """
        Configures the spline so that it corresponds to simplest straight 
        movement.
        """
        pass

    def _generate_random_trajectory(self, points):
        """
        Configures the spline so that it defines some random trajectory.
        """
        pass

    def _quality_along_trajectory(self, step):
        """
        Calculates the quality of current trajectory (defined by spline).
        """
        pass

    def train_trajectory(self, points = 100, step = None):
        """
        Trains the spline so that it has the best quality.
        """
        if step is None:
            step = self._qfb.w / points / 10
        pass

    def f(self, x):
        """
        Returns the values of trajectory function.
        """
        pass

    def save_to_file(self, filename):
        """
        Saves an image of the terrain combined with the trajectory curves.
        """
        pass

def main(filename):
    qfb = QualityFunctionBuilder()
    qfb.load_from_image(filename)
    car = CarBuilder()

    vtb = VehicleTrajectoryBuilder(qfb, car)
    vtb.train_trajectory()

    f = filename.split('.')
    f.insert(-1, 'solved')
    new_filename = '.'.join(f)
    vtb.save_to_file(new_filename)

class Tester:
    """
    Tester class.
    Contains methods for testing purposes.
    """
    def _log(self, message):
        print message
        sys.stdout.flush()

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

    def test_image_loading(self, filename = 'samples/2_holes.png'):
        qfb = QualityFunctionBuilder()
        qfb.load_from_image(filename)
        print qfb.w, qfb.h
        print qfb._im[0][0], qfb._im[400][100]
        print qfb._im[220][170], qfb._im[625][50]

    def test_Q_calculation(self, filename = 'samples/2_holes.png'):
        qfb = QualityFunctionBuilder()
        qfb.load_from_image(filename)
        qfb.set_custom_terrain_size((2403, 1993))
        axis_x = np.linspace(0, 2403, 240)
        axis_y = np.linspace(0, 1993, 199)
        img = [[qfb.Q(x, y) for x in axis_x] for y in axis_y]
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    # Tester().test_spline_builder()
    # Tester().test_image_loading()
    # Tester().test_Q_calculation()
    main('samples/2_holes.png')































