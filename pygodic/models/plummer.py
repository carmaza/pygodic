# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np

from .spherically_symmetric import SphericallySymmetric


class Plummer(SphericallySymmetric):
    """
    Plummer's model.

    """

    @property
    def name(cls):
        return "Plummer"

    @property
    def has_analytic_df(cls):
        return True

    def mass_density(self, r):
        """
        The radial profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        Mass density in units of $M/R^3$.

        """
        return 3. / (4. * np.pi * np.power(1. + r * r, 2.5))

    def deriv_mass_density(self, r):
        """
        The radial derivative of the profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        Radial derivative of the mass density in units of $M/R^4$.

        """
        return -15. * r / (4. * np.pi * np.power(1. + r * r, 3.5))

    def relative_potential(self, r, offset=0.):
        """
        The radial profile of the relative potential.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        Relative potential in units of $GM/R$.

        """
        return 1. / np.sqrt(1. + r * r)

    def deriv_relative_potential(self, r, offset=0.):
        """
        The radial derivative of the profile of the relative potential.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        Radial derivative of the relative potential in units of $GM/R^2$.

        """
        return -r / np.power(1. + r * r, 1.5)

    def drho_dpsi(self, r, offset=0.):
        """
        The ratio `deriv_mass_density(r)` / `deriv_relative_potential(r)`.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        The ratio of derivatives in units of $1/GR^2$.

        """
        return 15. / (4. * np.pi * (1. + r * r)**2.)

    def antideriv_df(self, e):
        """
        The antiderivative of the distribution function.

        Parameters
        ----------

        `e` : array_like
        Relative energy.

        Returns
        -------

        out : ndarray
        The antiderivative of the DF evaluated at the given energy.

        """
        n = 5
        F = 24. * np.sqrt(2.) / (7. * np.pi**3.)
        return F * np.power(e, n - 0.5) / (n - 0.5)

    def df(self, e):
        """
        The distribution function.

        Parameters
        ----------

        `e` : array_like
        Relative energy.

        Returns
        -------

        out : ndarray
        The DF evaluated at the given energy.

        """
        n = 5
        F = 24. * np.sqrt(2.) / (7. * np.pi**3.)
        return F * np.power(e, n - 1.5)
