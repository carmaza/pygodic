# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class :class:`.LogRhoVsLogPsi`.

"""

import numpy as np
import numpy.typing as npt

from scipy.interpolate import Akima1DInterpolator


# Akima1DInterpolator has abstract methods inherited from its parent class
# which make no sense for it, so opt not to override them. Details at
# https://github.com/scipy/scipy/blob/v1.10.1/scipy/interpolate/_cubic.py#L367-L465
class LogRhoVsLogPsi(Akima1DInterpolator):  # pylint: disable=abstract-method
    """
    Interpolant for the log-log mass density vs relative potential relationship.

    Parameters
    ----------

    model : obj
        The spherical model used to construct the interpolant. Must be a
        :class:`.SphericallySymmetric` instance.

    r_bounds : tuple
        The bounds of the radial grid on which to evaluate the fields.

    pts_rad : int
        The target number of points of the radial grid.

    Notes
    -----

    - This class uses Akima's 1D interpolator (1970). This choice is justified,
      as we obtain the interpolation points to arbitrary precision from
      analytic profiles.

    - This class inherits from scipy.interpolate.Akima1DInterpolator_.

    .. _scipy.interpolate.Akima1DInterpolator: https://docs.scipy.org/doc/scipy
       /reference/generated/scipy.interpolate.Akima1DInterpolator.html

    """

    def __init__(self, model, r_bounds, pts_rad):
        self._radial_grid = np.linspace(r_bounds[0], r_bounds[1], pts_rad)

        # Radial profiles are decreasing so we flip arrays to interpolate.
        rev_grid = np.flip(self._radial_grid)
        psi = model.relative_potential(rev_grid)
        rho = model.mass_density(rev_grid)

        logpsi = np.log10(psi)
        self._logrho = np.log10(rho)
        self._dlr_dlp = psi * model.drho_dpsi(rev_grid) / rho

        super().__init__(logpsi, self._logrho)

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
        return self.x

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
