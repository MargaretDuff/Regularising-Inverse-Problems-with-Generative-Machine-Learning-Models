# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:54:58 2020

@author: marga
"""
import numpy as np
import odl
from scipy.ndimage import convolve as sp_convolve

class Convolution(odl.Operator):

    def __init__(self, space, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.size/ len(kernel)

        super().__init__(space, space, linear=True)

    def _call(self, x, out):
        sp_convolve(x, self.kernel, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            if self.domain.ndim == 2:
                kernel = np.fliplr(np.flipud(self.kernel.copy().conj()))
                kernel = self.kernel.space.element(kernel)
            else:
                raise NotImplementedError('"adjoint_kernel" only defined for '
                                          '2d kernels')

            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = Convolution(self.domain, kernel, origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)
