# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class :class:`AntiderivDfVsEnergy`

"""

import numpy as np

from scipy.interpolate import CubicSpline

from pygodic import eddington_inversion


class AntiderivDfVsEnergy(CubicSpline):
    """
    Interpolant for the antiderivative of the DF vs relative energy.

    Parameters
    ----------

    `lrlp` : obj
        The B-spline representation of the log-log derivative of the density
        with respect to the potential. Must be a

    `n_pts` : int
    The number of points used to obtain the interpolant.

    `k` : int (optional, default: 3)
    The order of the interpolation. Must be in the range [1, 5].

    """
    def __init__(self, lrlp):
        
    energy = np.geomspace(psi_min, psi_max, n_pts)
    antideriv_df = eddington_inversion.antideriv_df(energy, psi_min,
                                                    drho_dpsi_spline)

    antideriv_df_spline = interpolate.spline_representation(energy,
                                                            antideriv_df,
                                                            k=k)

    return energy, antideriv_df_spline
