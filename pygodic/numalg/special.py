# Distributed under the MIT License.
# See LICENSE for details.
"""
Wraps some functions from `scipy.special`.

"""

import scipy.special


def roots_jacobi(n, alpha, beta, mu=False): # pylint: disable=invalid-name
    """
    Wrapper to `scipy.special.roots_jacobi`. See the SciPy documentation
    for details.

    """
    return scipy.special.roots_jacobi(n, alpha, beta, mu)
