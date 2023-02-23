# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class ``LogRhoVsLogPsi``.

"""

import numpy as np
import numpy.typing as npt

from scipy.interpolate import UnivariateSpline


class LogRhoVsLogPsi(UnivariateSpline):
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

    This class inherits from ``scipy.interpolate.UnivariateSpline``. See
    `its documentation`_ for details.

    .. _its documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html

    """

    def __init__(self, model, r_bounds, pts_rad, k=3):
        self._radial_grid = np.linspace(r_bounds[0], r_bounds[1], pts_rad)

        # Radial profiles are decreasing so we flip arrays to interpolate.
        psi = np.flip(model.relative_potential(self._radial_grid))
        rho = np.flip(model.mass_density(self._radial_grid))

        self._logpsi = np.log10(psi)
        self._logrho = np.log10(rho)
        self._dlr_dlp = psi * np.flip(model.drho_dpsi(self._radial_grid)) / rho

        super().__init__(self._logpsi,
                         self._logrho,
                         k=k,
                         ext='raise',
                         check_finite=True)

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
