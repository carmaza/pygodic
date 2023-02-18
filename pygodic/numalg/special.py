# Distributed under the MIT License.
# See LICENSE for details.
"""
Wraps some functions from `scipy.special`.

"""

import scipy.special


def roots_jacobi(n, alpha, beta, mu=False):  # pylint: disable=invalid-name
    """
    Wrapper to `scipy.special.roots_jacobi`. See the SciPy documentation
    for details.

    """
    return scipy.special.roots_jacobi(n, alpha, beta, mu)


def erf(z, out=None):
    """
    Wrapper to `scipy.special.erf`. See the SciPy documentation for details.

    """
    return scipy.special.erf(z, out)  # pylint: disable=no-member


def erfi(z, out=None):
    """
    Wrapper to `scipy.special.erfi`. See the SciPy documentation for details.

    """
    return scipy.special.erfi(z, out)  # pylint: disable=no-member
