# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class ``LogRhoVsLogPsi``.

"""

import numpy as np
import numpy.typing as npt

from scipy.interpolate import Akima1DInterpolator


class LogRhoVsLogPsi(Akima1DInterpolator):
    """
    Interpolant for the log-log mass density vs relative potential relationship.

    Parameters
    ----------

    model : obj
        The spherical model used to construct the interpolant. Must be a
        SphericalModel instance.

    r_bounds : tuple
        The bounds of the radial grid on which to evaluate the fields.

    pts_rad : int
        The target number of points of the radial grid.

    k : int (optional, default: 3)
        The degree of the smoothing spline. Must be one of ``[1, 2, 3, 4, 5]``.
        `k = 3` is a cubic spline.

    Notes
    -----

    - This class uses Akima's 1D interpolator (1970). This choice is justified,
      as we obtain the interpolation points to arbitrary precision from
      analytic profiles.

    - This class inherits from ``scipy.interpolate.Akima1DInterpolator``. See
      `its documentation`_ for a reference on its additional members.

    .. _its documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html

    """

    def __init__(self, model, r_bounds, pts_rad, k=3):
        self._radial_grid = np.linspace(r_bounds[0], r_bounds[1], pts_rad)

        # Radial profiles are decreasing so we flip arrays to interpolate.
        rev_grid = np.flip(self._radial_grid)
        psi = model.relative_potential(rev_grid)
        rho = model.mass_density(rev_grid)

        self._logpsi = np.log10(psi)
        self._logrho = np.log10(rho)
        self._dlr_dlp = psi * model.drho_dpsi(rev_grid) / rho

        super().__init__(self._logpsi, self._logrho)

    @property
    def dlr_dlp(self) -> npt.NDArray:
        """
        The derivative of log density with respect to log potential.

        """
        return self._dlr_dlp

    @property
    def logpsi(self) -> npt.NDArray:
        """
        The logarithm of the relative potential.

        """
        return self._logpsi

    @property
    def logrho(self) -> npt.NDArray:
        """
        The logarithm of the mass density.

        """
        return self._logrho

    @property
    def radial_grid(self) -> npt.NDArray:
        """
        The radial grid on which to evaluate the fields.

        """
        return self._radial_grid
