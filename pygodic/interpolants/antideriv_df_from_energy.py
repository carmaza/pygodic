# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines `interpolants.antideriv_df_from_energy`, which computes the interpolant
for the antiderivative of the DF as a function of the energy.

"""

import numpy as np

from pygodic import eddington_inversion
from pygodic.numalg import interpolate


def antideriv_df_from_energy(psi_min, psi_max, drho_dpsi_spline, n_pts, k=3):
    """
    For the given model, get interpolant for the antiderivative of the DF as a
    function of the relative energy.

    Details
    -------

    The interpolant will be a B-spline of the given order `k`. The absisas of
    the interpolation will be a log-spaced set of energies bounded by `psi_min`
    and `psi_max`. The ordinates of the interpolation will be the set returned
    by the routine `eddington_inversion.antideriv_df`.

    Parameters
    ----------

    `psi_min`, `psi_max` : float, float
    The bounds of the energy range within which the interpolant will be valid.

    `drho_dpsi_spline` : tuple
    The B-spline representation of the derivative of the mass density in the
    range of potentials bounded by `psi_min` and `psi_max`.

    `n_pts` : int
    The number of points used to obtain the interpolant.

    `k` : int (optional, default: 3)
    The order of the interpolation. Must be in the range [1, 5].

    Returns
    -------

    out : ndarray, object
    Tuple containing, respectively, the set of energies used to obtain the
    interpolant, and the B-spline representation of the antiderivative of the
    DF.

    """
    energy = np.geomspace(psi_min, psi_max, n_pts)
    antideriv_df = eddington_inversion.antideriv_df(energy, psi_min,
                                                    drho_dpsi_spline)

    antideriv_df_spline = interpolate.spline_representation(energy,
                                                            antideriv_df,
                                                            k=k)

    return energy, antideriv_df_spline
