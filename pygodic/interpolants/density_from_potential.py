# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines `interpolants.DensityFromPotential`.

"""

import numpy as np

from scipy.interpolate import UnivariateSpline


class LogRhoVsLogPsi(UnivariateSpline):
    """
    Interpolant for the log-log mass density vs relative potential relationship.

    Notes
    -----

    This class inherits from `scipy.interpolate.UnivariateSpline`. See the SciPy
    documentation for details.

    """

    def __init__(self, model, r_bounds, pts_rad, k=3):
        """
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
            The degree of the smoothing spline. Must be one of [1, 2, 3, 4, 5].
            k = 3 is a cubic spline.

        """
        self._radial_grid = np.linspace(r_bounds[0], r_bounds[1], pts_rad)

        # Radial profiles are decreasing so we need to flip arrays to interpolate.
        self._logpsi = np.log10(
            np.flip(model.relative_potential(self._radial_grid)))
        self._logrho = np.log10(np.flip(model.mass_density(self._radial_grid)))

        super().__init__(self._logpsi,
                         self._logrho,
                         k=k,
                         ext='raise',
                         check_finite=True)

    @property
    def radial_grid(self):
        """
        The radial grid on which to evaluate the fields.

        """
        return self._radial_grid

    @property
    def logpsi(self):
        """
        The logarithm of the relative potential.

        """
        return self._logpsi

    @property
    def logrho(self):
        """
        The logarithm of the mass density.

        """
        return self._logrho
