# Distributed under the MIT License.
# See LICENSE for details.
"""
Wraps some functions from `scipy.interpolate`.

"""

import scipy.interpolate


def spline_representation(x,
                          y,
                          w=None, # pylint: disable=invalid-name
                          xb=None, # pylint: disable=invalid-name
                          xe=None, # pylint: disable=invalid-name
                          k=3,
                          task=0,
                          s=None, # pylint: disable=invalid-name
                          t=None, # pylint: disable=invalid-name
                          full_output=0,
                          per=0,
                          quiet=1):
    """
    Wrapper to `scipy.interpolate.splrep`. See the SciPy documentation for
    details.

    """
    return scipy.interpolate.splrep(x, y, w, xb, xe, k, task, s, t,
                                    full_output, per, quiet)


def spline_evaluation(x, tck, der=0, ext=0):
    """
    Wrapper to `scipy.interpolate.splev`. See the SciPy documentation for
    details.

    """
    return scipy.interpolate.splev(x, tck, der, ext)
