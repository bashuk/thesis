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
import math
import profilehooks

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
        # Spline miles
        self.miles = 0
        self.mile_length = 0
        self.mile = []

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
            index = self._n - 1
        else:
            index = bisect.bisect_right(self._x, x) - 1

        x0 = self._x[index]
        a = self._a[index]
        b = self._b[index]
        c = self._c[index]
        d = self._d[index]
        value = float(a + b * (x - x0) + c * (x - x0) ** 2 + d * (x - x0) ** 3)

        return value

    def der_f(self, x):
        """
        Calculates the derivative of spline approximation in a given point.
        """
        if self._n == 0:
            raise Exception("Spline not built yet.")
        if x < self._x[0] or self._x[-1] < x:
            raise ValueError("Given point is out of interval.")

        x = float(x)

        if x == self._x[-1]:
            index = self._n - 1
        else:
            index = bisect.bisect_right(self._x, x) - 1

        x0 = self._x[index]
        a = self._a[index]
        b = self._b[index]
        c = self._c[index]
        d = self._d[index]
        der = b + 2.0 * c * (x - x0) + 3.0 * d * (x - x0) ** 2

        return der

    def split_by_length(self, miles, integration_pieces_per_mile = 10):
        """
        Splits calculated spline into m pieces, equal by lenght.
        """
        M = miles
        N = miles * integration_pieces_per_mile

        x = map(float, np.linspace(self._x[0], self._x[-1], 2 * N + 1))
        y = map(lambda x: math.sqrt(1.0 + self.der_f(x) ** 2), x)

        l = [0.0]
        for i in range(N):
            intgr = (x[2 * i + 2] - x[2 * i]) / 6.0 * \
                (y[2 * i] + 4.0 * y[2 * i + 1] + y[2 * i + 2])
            l.append(intgr)
        for i in range(1, len(l)):
            l[i] = l[i] + l[i - 1]

        mile_length = l[-1] / M
        self.mile = [0.0]
        wanted = mile_length
        for i in range(N):
            if l[i] <= wanted and wanted < l[i + 1]:
                # alpha == 1.0 means left, alpha == 0.0 means right
                alpha = (l[i + 1] - wanted) / (l[i + 1] - l[i])
                self.mile.append(
                    alpha * x[2 * i] + (1.0 - alpha) * x[2 * (i + 1)])
                wanted += mile_length
        if len(self.mile) == M:
            self.mile.append(x[-1])
        self.miles = M
        self.mile_length = mile_length

        if len(self.mile) != M + 1:
            raise Exception("Ooops... I think we didn't manage to split it.")

    def show(self):
        """
        Show spline.
        """
        x = np.linspace(self._x[0], self._x[-1], 400)
        y = [self.f(xi) for xi in x]

        dot_x = self._x
        dot_y = [self.f(xi) for xi in dot_x]

        plt.plot(x, y, 'g', dot_x, dot_y, 'ro')
        plt.show()

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
            [float(pix[x, y]) / 255.0 for y in xrange(self._imh)] 
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

        # Going beyond the boundaries of the road line is not acceptable.
        if y < 0 or y > self.h:
            # I thought of the exception here at first, but then I realised
            # that splines can go beyond the road, and when calculating
            # the quality for that spline – we don't want it to raise any
            # exceptions, we just want it to make an enormously big jump
            return - 10 ** 9
            # raise ValueError("Given point is out of the terrain.")
        # Since we have no info about the quality of the road there, we just
        # have to assume that those values are the same as on the boundaries.
        if x < 0:
            x = 0
        if x > self.w:
            x = self.w

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
    def __init__(self, width = 171, length = 271, wheel = 31.5):
        """
        Default parameters (width = 171, length = 271, wheel = 315) are
        the actual parameters of Bugatti Veyron 16.4, given in cm.
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
        # Helper classes
        self._qfb = qfb
        self._car = car
        self._sb = SplineBuilder()
        # Terrain parameters
        self._w = qfb.w
        self._h = qfb.h
        # Boundaries
        self._fl = fl if fl is not None else qfb.h * 0.5
        self._fr = fr if fr is not None else qfb.h * 0.5
        self._dfl = dfl if dfl is not None else 0.0
        self._dfr = dfr if dfr is not None else 0.0
        # Quality along trajectory
        self._qat = - 10 ** 9
        # Trained flag
        self._trained = False
        # wheels traces
        self._rlw = []
        self._rrw = []
        self._flw = []
        self._frw = []

    def _generate_straight_trajectory(self, points, miles_per_point = 10,
        rebuild_spline = True):
        """
        Configures the spline so that it corresponds to simplest straight 
        movement.
        """
        x = map(float, np.linspace(0.0, self._w, points + 1))
        y = [self._h / 2.0 for i in xrange(points + 1)]
        y[0] = self._fl
        y[points] = self._fr

        if rebuild_spline:
            self._sb.build(x, y, self._dfl, self._dfr)
            miles = points * miles_per_point
            self._qat = self._quality_along_trajectory(miles)

        return x, y

    def _generate_random_trajectory(self, points, miles_per_point = 10,
        rebuild_spline = True):
        """
        Configures the spline so that it defines some random trajectory.
        """
        margin = (self._car.width + self._car.wheel) / 2.0
        x = map(float, np.linspace(0.0, self._w, points + 1))
        y = [random.uniform(0.0 + margin, self._h - margin) 
            for i in xrange(points + 1)]
        y[0] = self._fl
        y[points] = self._fr
        
        if rebuild_spline:
            self._sb.build(x, y, self._dfl, self._dfr)
            miles = points * miles_per_point
            self._qat = self._quality_along_trajectory(miles)

        return x, y

    def _try_alternative_trajectory(self, new_x, new_y, miles, force = False):
        """
        Tries alternative trajectory.
        If the quality along the new trajectory is higher, or if forced,
        the spline is rebuilt along the new trajectory.
        Otherwise, the spline is not changed.
        Return True if the spline was changed, False otherwise.
        """
        cur_x = copy.deepcopy(self._sb._x)
        cur_y = copy.deepcopy(self._sb._y)

        self._sb.build(new_x, new_y, self._dfl, self._dfr)
        new_qat = self._quality_along_trajectory(miles)
        if new_qat > self._qat or force:
            self._qat = new_qat
            return True
        else:
            self._sb.build(cur_x, cur_y, self._dfl, self._dfr)
            return False

    def _quality_along_trajectory(self, miles, save_trace = False):
        """
        Calculates the quality of current trajectory (defined by spline).
        """
        self._sb.split_by_length(miles)

        # wheels traces
        self._rlw = []
        self._rrw = []
        self._flw = []
        self._frw = []

        # Q1 is integral over quality function. 
        # This corresponds to quality of the road. 
        # The more - the better.
        # Implemented with the method of right rectangles
        Q1 = 0.0
        for i in xrange(1, len(self._sb.mile)):
            # print 'i', i # debug
            # print 'mile', self._sb.mile_length # debug
            # Start and end points of the arc
            x1 = self._sb.mile[i - 1]
            y1 = self._sb.f(x1)
            x2 = self._sb.mile[i]
            y2 = self._sb.f(x2)

            # Derivatives on the ends, corresponding angles
            d1 = self._sb.der_f(x1)
            a1 = math.atan(d1)
            d2 = self._sb.der_f(x2)
            a2 = math.atan(d2)
            # print 'xyda', x1, y1, d1, a1 # debug
            # print 'xyda', x2, y2, d2, a2 # debug

            # Additional vectors for easy computation of coordinates of
            # all 4 wheels
            start = np.array([x1, y1])
            finish = np.array([x2, y2])
            front_move = (self._car.length) * np.array([np.cos(a1), np.sin(a1)])
            wheel_move = (self._car.width * 0.5) * np.array(
                    [np.cos(a1 + np.pi * 0.5), np.sin(a1 + np.pi * 0.5)])
            # print 'st fi', start, finish # debug
            # print 'frmv whmv', front_move, wheel_move # debug

            if abs(a1 - a2) < 1e-5:
                # print 'Case 1' # debug
                # Case 1: line
                # All wheels will cover the same area in this case.
                # The only difference is where we take the average quality -
                # this point will be individual for each wheel.
                area = self._car.wheel * self._sb.mile_length
                mile_move = (self._sb.mile_length * 0.5) * \
                    np.array([np.cos(a1), np.sin(a1)])
                # print 'area', area # debug
                # print 'mimv', mile_move # debug

                # Rear wheels
                rear = start + mile_move # rear suspention center
                lw = rear + wheel_move
                Q1 += area * self._qfb.Q(*lw) # left wheel
                self._rlw.append(lw)

                rw = rear - wheel_move
                Q1 += area * self._qfb.Q(*lw) # right wheel
                self._rrw.append(rw)
                # print 'lw rw', lw, rw # debug

                # Front wheels
                front = start + mile_move + front_move # front suspention center
                lw = front + wheel_move
                Q1 += area * self._qfb.Q(*lw) # left wheel
                self._flw.append(lw)

                rw = front - wheel_move
                Q1 += area * self._qfb.Q(*lw) # right wheel
                self._frw.append(rw)
                # print 'lw rw', lw, rw # debug
            else:
                # print 'Case 2' # debug
                # Case 2: arc
                # Wheels are moving according to the Ackermann steering.
                
                # Finding parameters of the arc
                # Radial lines
                a1p = a1 + np.pi * 0.5
                l1a = - np.sin(a1p) # = y1 - (y1 + np.sin(a1p))
                l1b = np.cos(a1p) # = (x1 + np.cos(a1p)) - x1
                l1c = - (l1a * x1 + l1b * y1)

                a2p = a2 + np.pi * 0.5
                l2a = - np.sin(a2p) # = y2 - (y2 + np.sin(a2p))
                l2b = np.cos(a2p) # = (x2 + np.cos(a2p)) - x2
                l2c = - (l2a * x2 + l2b * y2)

                # Center is the intersection of the lines
                cx = float((l1b * l2c - l2b * l1c) / (l1a * l2b - l2a * l1b))
                cy = float((l1c * l2a - l2c * l1a) / (l1a * l2b - l2a * l1b))
                co = np.array([cx, cy])

                # Arc radius
                # Not always there is an arc that perfectly fits the given
                # boundary conditions (x, y, a). That's why this circular
                # arc is just an approximation.
                R1 = np.linalg.norm(start - co)
                R2 = np.linalg.norm(finish - co)
                cr = float(R1 + R2) / 2.0

                # Arc radial size
                v1 = start - co # radial vector
                v2 = finish - co # radial vector
                v1u = v1 / np.linalg.norm(v1) # unit vector
                v2u = v2 / np.linalg.norm(v2) # unit vector
                ca = np.arccos(np.dot(v1u, v2u)) # circular arc angle
                if np.isnan(ca):
                    if (v1u == v2u).all():
                        return 0.0
                    else:
                        return np.pi
                ca = float(ca)

                # Orienting the angle
                # http://e-maxx.ru/algo/oriented_area
                s = (cx - x1) * (y2 - y1) - (cy - y1) * (x2 - x1)
                if s > 0:
                    # counter-clockwise order - angle must be negative
                    ca = - ca
                # print 'co cr ca', co, cr, ca # debug

                # Rotation matrix (rotates the half of the arc angle)
                rot_a = ca * 0.5
                rot_m = np.array([[np.cos(rot_a), - np.sin(rot_a)], 
                    [np.sin(rot_a), np.cos(rot_a)]])
                # print 'rot_m', rot_m # debug

                # Rear left wheel
                start_w = start + wheel_move
                rel_start_w = start_w - co
                rel_mid_w = np.dot(rot_m, rel_start_w)
                inner_r = np.linalg.norm(rel_mid_w) - self._car.wheel * 0.5
                outer_r = inner_r + self._car.wheel
                area = (outer_r ** 2 - inner_r ** 2) * abs(ca) / 2.0
                mid_w = rel_mid_w + co
                Q1 += float(area * self._qfb.Q(*mid_w))
                self._rlw.append(mid_w)
                # print 'rl: area midw', area, mid_w # debug

                # Rear right wheel
                start_w = start - wheel_move
                rel_start_w = start_w - co
                rel_mid_w = np.dot(rot_m, rel_start_w)
                inner_r = np.linalg.norm(rel_mid_w) - self._car.wheel * 0.5
                outer_r = inner_r + self._car.wheel
                area = (outer_r ** 2 - inner_r ** 2) * abs(ca) / 2.0
                mid_w = rel_mid_w + co
                Q1 += float(area * self._qfb.Q(*mid_w))
                self._rrw.append(mid_w)
                # print 'rr: area midw', area, mid_w # debug

                # Front left wheel
                start_w = start + front_move + wheel_move
                rel_start_w = start_w - co
                rel_mid_w = np.dot(rot_m, rel_start_w)
                inner_r = np.linalg.norm(rel_mid_w) - self._car.wheel * 0.5
                outer_r = inner_r + self._car.wheel
                area = (outer_r ** 2 - inner_r ** 2) * abs(ca) / 2.0
                mid_w = rel_mid_w + co
                Q1 += float(area * self._qfb.Q(*mid_w))
                self._flw.append(mid_w)
                # print 'fl: area midw', area, mid_w # debug

                # Front right wheel
                start_w = start + front_move - wheel_move
                rel_start_w = start_w - co
                rel_mid_w = np.dot(rot_m, rel_start_w)
                inner_r = np.linalg.norm(rel_mid_w) - self._car.wheel * 0.5
                outer_r = inner_r + self._car.wheel
                area = (outer_r ** 2 - inner_r ** 2) * abs(ca) / 2.0
                mid_w = rel_mid_w + co
                Q1 += float(area * self._qfb.Q(*mid_w))
                self._frw.append(mid_w)
                # print 'fr: area midw', area, mid_w # debug
            # print 'Q1', Q1 # debug
            # print '----------------------------' # debug
            # raw_input() # debug

        # TODO 1: this is no longer valid, update.
        # Q2 is the integral over constant scalar field.
        # This corresponds to the length of the trajectory. 
        # The less - the better.
        Q2 = miles * self._sb.mile_length

        # Total quality along the trajectory.
        # The more - the better.
        # Q1: [0.0 .. 4 * wheel * w]
        # Q2: [qfb.w .. sinusoidal_length]
        res = Q1 / (4.0 * self._qfb.w * self._car.wheel) * 100.0 \
            - Q2 / self._qfb.w * 1.0

        if not(save_trace):
            self._rlw = []
            self._rrw = []
            self._flw = []
            self._frw = []

        return res

    def train_trajectory(self, points = 10, miles_per_point = 10):
        """
        Trains the spline so that it has the best quality.
        """
        # TODO 1: train softly
        miles = points * miles_per_point

        # Choosing from given number of random trajectories
        self._generate_straight_trajectory(points, miles_per_point)
        x = copy.deepcopy(self._sb._x)
        best_qat = self._qat
        best_y = copy.deepcopy(self._sb._y)

        for attempt in xrange(10):
            # These two parameters define the process of optimization
            # Also, they define the number of iterations
            jump_step = self._qfb.h
            threshold = 1.0

            self._generate_random_trajectory(points, miles_per_point)
            while jump_step > threshold:
                for point in xrange(1, points):
                    y = copy.deepcopy(self._sb._y)
                    init_point_y = y[point]

                    # trying to jump up
                    y[point] = init_point_y + jump_step
                    self._try_alternative_trajectory(x, y, miles)
                    # trying to jump down
                    y[point] = init_point_y - jump_step
                    self._try_alternative_trajectory(x, y, miles)

                    sys.stdout.write(
                        "\rAttempt #{}: jump = {}, quality = {}".format(
                        attempt + 1, jump_step, self._qat))
                    sys.stdout.flush()
                    # self.show()
                jump_step *= 0.5
            sys.stdout.write("\n")

            if self._qat > best_qat:
                best_qat = self._qat
                best_y = copy.deepcopy(self._sb._y)

        self._sb.build(x, best_y, self._dfl, self._dfr)
        self._qat = self._quality_along_trajectory(miles, save_trace = True)
        self._trained = True
        print 'Best: {}'.format(best_qat)

    def f(self, x):
        """
        Returns the values of trajectory function.
        """
        return self._sb.f(x)

    def Q(self, x, y):
        """
        Returns the values of quality function.
        """
        return self._qfb.Q(x, y)

    def show(self):
        """
        Show quality function and current trajectory.
        """
        Q_x = np.linspace(0, self._qfb.w, self._qfb.w)
        Q_y = np.linspace(0, self._qfb.h, self._qfb.h)
        img = [[self.Q(x, y) for x in Q_x] for y in Q_y]
        plt.imshow(img)

        f_x = np.linspace(0, self._qfb.w, self._qfb.w)
        f_y = [self.f(xi) for xi in f_x]
        plt.plot(f_x, f_y, 'g', label='Trajectory (spline)')

        plt.plot(self._sb._x, self._sb._y, 'bo', label='Key points')

        # wheels traces
        rl_x = [w[0] for w in self._rlw]
        rl_y = [w[1] for w in self._rlw]
        plt.plot(rl_x, rl_y, 'yo')
        rr_x = [w[0] for w in self._rrw]
        rr_y = [w[1] for w in self._rrw]
        plt.plot(rr_x, rr_y, 'yo')
        fl_x = [w[0] for w in self._flw]
        fl_y = [w[1] for w in self._flw]
        plt.plot(fl_x, fl_y, 'co')
        fr_x = [w[0] for w in self._frw]
        fr_y = [w[1] for w in self._frw]
        plt.plot(fr_x, fr_y, 'co')

        plt.title('Optimal trajectory')
        plt.axis([Q_x[0], Q_x[-1], Q_y[0], Q_y[-1]])
        plt.show()

    def save_to_file(self, filename, scale = 1):
        """
        Saves an image of the terrain combined with the trajectory curves.
        """
        # TODO 2: draw all 4 wheels
        w, h = int(self._w * scale), int(self._h * scale)
        im = PIL.Image.new("RGB", (w, h))
        pix = im.load()

        # Drawing the quality of the terrain
        for i in xrange(w):
            for j in xrange(h):
                px, py = i * 1.0 / scale, j * 1.0 / scale
                R, G, B = [int(self._qfb.Q(px, py) * 255.0)] * 3
                pix[i, j] = (R, G, B)

        # Drawing the trajectory
        alpha = 0.7
        for i in xrange(w):
            x = i * 1.0 / scale
            y = self._sb.f(x)

            j0 = int(y * scale)
            low = max(0, j0)
            high = min(h, j0 + 1)
            for j in range(low, high):
                R, G, B = pix[i, j]
                R = int(alpha * 0   + (1.0 - alpha) * R)
                G = int(alpha * 255 + (1.0 - alpha) * G)
                B = int(alpha * 0   + (1.0 - alpha) * B)
                pix[i, j] = (R, G, B)

        im.save(filename)

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
        sb.build(sub_x, sub_y, 1.0, 1.0)
        z = [sb.f(point) for point in x]

        plt.plot(x, y, 'b--', sub_x, sub_y, 'ro', x, z, 'g')
        plt.show()

    def test_spline_split_by_length(self, split_pieces = 5):
        x = np.linspace(0, 2 * np.pi, 500)
        y = np.sin(x)

        sb = SplineBuilder()
        sb.build(x, y, 1.0, 1.0)
        z = [sb.f(point) for point in x]

        sb.split_by_length(split_pieces)
        mx = sb.mile
        my = np.sin(mx)

        plt.plot(x, y, 'b--', x, z, 'g', mx, my, 'ro')
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

    def test_trajectory_drawing(self, filename = 'samples/2_holes.png'):
        qfb = QualityFunctionBuilder()
        qfb.load_from_image(filename)
        car = CarBuilder()
        vtb = VehicleTrajectoryBuilder(qfb, car)
        vtb._generate_random_trajectory(10)
        vtb.save_to_file('samples/result.png')

    def test_quality_along_trajectory(self):
        qfb = QualityFunctionBuilder()
        qfb.load_from_image('samples/2_holes.png')
        qfb.set_custom_terrain_size((1500, 300))
        car = CarBuilder()
        vtb = VehicleTrajectoryBuilder(qfb, car)

        x, y = vtb._generate_straight_trajectory(10, rebuild_spline = False)
        y[1] += 0.0
        y[2] += 5.0
        y[3] += 10.0
        y[4] += 15.0
        y[5] += 15.0
        y[6] += 15.0
        y[7] += 15.0
        y[8] += 10.0
        y[9] += 5.0
        vtb._sb.build(x, y)
        print vtb._quality_along_trajectory(100)
        
        vtb.show()

    # @profilehooks.profile
    def test_train_trajectory(self):
        qfb = QualityFunctionBuilder()
        qfb.load_from_image('samples/2_holes.png')
        qfb.set_custom_terrain_size((1500, 300))
        car = CarBuilder()
        vtb = VehicleTrajectoryBuilder(qfb, car)
        
        vtb.train_trajectory(10)
        vtb.show()
        # vtb.save_to_file('samples/result.png')

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

if __name__ == '__main__':
    # print "Python loaded. Let's rock!"
    # Tester().test_spline_builder()
    # Tester().test_spline_split_by_length(5)
    # Tester().test_image_loading()
    # Tester().test_Q_calculation()
    # Tester().test_trajectory_drawing()
    # Tester().test_quality_along_trajectory()
    Tester().test_train_trajectory()

    # main('samples/2_holes.png')
    pass

# TODO 3: remove #debug lines
# TODO 3: implement various checks and validations for ALL methods
# TODO 3: make global constants (parameters of the algo)
# TODO 3: add information messages to console
# TODO 3: optimize code
























