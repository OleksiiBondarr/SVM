import numpy as np
import numpy as np

import numpy.linalg as la





class Kernel(object):

    """Implements list of kernels from

    http://en.wikipedia.org/wiki/Support_vector_machine

    """

    @staticmethod

    def linear():

        def f(x, y):

            return np.inner(x, y)

        return f



    @staticmethod

    def gaussian(sigma):

        def f(x, y):

            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))

            return np.exp(exponent)

        return f


