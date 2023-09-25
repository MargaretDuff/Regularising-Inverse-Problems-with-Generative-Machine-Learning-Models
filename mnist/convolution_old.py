# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:36:43 2020

@author: marga
"""

import scipy.signal
import odl 

class Convolution(odl.Operator):
    """Operator calculating the convolution of a kernel with a function.

    The operator inherits from ``odl.Operator`` to be able to be used with ODL.
    """

    def __init__(self, kernel):
        """Initialize a convolution operator with a known kernel."""

        # Store the kernel
        self.kernel = kernel

        # Initialize the Operator class by calling its __init__ method.
        # This sets properties such as domain and range and allows the other
        # operator convenience functions to work.
        super(Convolution, self).__init__(
            domain=kernel.space, range=kernel.space, linear=True)

    def _call(self, x):
        """Implement calling the operator by calling scipy."""
        return scipy.signal.fftconvolve(self.kernel, x, mode='same')
    
    @property  # making the adjoint a property lets users access it as conv.adjoint
    def adjoint(self):
        return self  # the adjoint is the same as this operator