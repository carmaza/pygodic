# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `models.Jaffe`.

"""

import numpy as np

from scipy import special

from .spherically_symmetric import SphericallySymmetric


class Jaffe(SphericallySymmetric):
    """
    Jaffe's model.

    """

    @classmethod
    def name(cls):
        """
        The class name.

        """
        return "Jaffe"

    @classmethod
    def has_analytic_df(cls):
        """
        Whether the model has an analytic DF.

        """
        return True

    def mass_density(self, r):
        """
        The radial profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$, the half-mass radius.

        Returns
        -------

        out : ndarray
        Mass density in units of $M/R^3$.

        """
        return 1. / (4. * np.pi * (r * (1. + r))**2.)

    def deriv_mass_density(self, r):
        """
        The radial derivative of the profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$, the half-mass radius.

        Returns
        -------

        out : ndarray
        Radial derivative of the mass density in units of $M/R^4$.

        """
        return -(1. + 2. * r) / (r * (1. + r))**3. / (2. * np.pi)

    def relative_potential(self, r, offset=0.):
        """
        The radial profile of the relative potential.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$, the half-mass radius.

        Returns
        -------

        out : ndarray
        Relative potential in units of $GM/R$.

        """
        return np.log((1. + r) / r)

    def deriv_relative_potential(self, r, offset=0.):
        """
        The radial derivative of the profile of the relative potential.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$, the half-mass radius.

        Returns
        -------

        out : ndarray
        Radial derivative of the relative potential in units of $GM/R^2$.

        """
        return -1. / (r * (1. + r))

    def drho_dpsi(self, r, offset=0.):
        """
        The ratio `deriv_mass_density(r)` / `deriv_relative_potential(r)`.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$, the half-mass radius.

        Returns
        -------

        out : ndarray
        The ratio of derivatives in units of $1/GR^2$.

        """
        return (1. + 2. * r) / (r * (1. + r))**2. / (2. * np.pi)

    @staticmethod
    def _f_pm(x, plus_or_minus):
        """
        Dawson's integral plus and minus in terms of erf and erfi functions.

        """
        if plus_or_minus == +1:
            return 0.5 * np.sqrt(np.pi) * np.exp(-x * x) * special.erfi(x)
        if plus_or_minus == -1:
            return 0.5 * np.sqrt(np.pi) * np.exp(x * x) * special.erf(x)
        raise ValueError(
            "In Dawson's Integral: value different from +1 or -1.")

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
        return (self._f_pm(np.sqrt(2. * e), -1) -
                self._f_pm(np.sqrt(2. * e), +1) + np.sqrt(8.) *
                (self._f_pm(np.sqrt(e), +1) - self._f_pm(np.sqrt(e), -1))) / (
                    4. * np.pi**3.)

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
        return (self._f_pm(np.sqrt(2 * e), -1) +
                self._f_pm(np.sqrt(2 * e), +1) - np.sqrt(2) *
                (self._f_pm(np.sqrt(e), -1) + self._f_pm(np.sqrt(e), +1))) / (
                    2. * np.pi**3.)
