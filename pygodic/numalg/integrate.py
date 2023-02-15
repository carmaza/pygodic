# Distributed under the MIT License.
# See LICENSE for details.
"""
Wraps some functions from `scipy.integrate`.

"""

import scipy.integrate


def nquad(func, ranges, args=None, opts=None, full_output=False):
    """
    Wrapper to `scipy.integrate.nquad`. See the SciPy documentation for details.

    """
    return scipy.integrate.nquad(func, ranges, args, opts, full_output)


def romberg(function,
            a,
            b,
            args=(),
            tol=1.e-12,
            rtol=1.e-12,
            show=False,
            divmax=10,
            vec_func=False):
    """
    Wrapper to `scipy.integrate.romberg`. See the SciPy documentation for
    details.

    """
    return scipy.integrate.romberg(function, a, b, args, tol, rtol, show,
                                   divmax, vec_func)


def simpson(y, x=None, dx=1.0, axis=-1, even='avg'):
    """
    Wrapper to `scipy.integrate.simpson`. See the SciPy documentation for
    details.

    """
    return scipy.integrate.simpson(y, x, dx, axis, even)
