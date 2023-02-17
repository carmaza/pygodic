# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `models.HenonIsochrone`.

"""

import numpy as np

from .spherically_symmetric import SphericallySymmetric


class HenonIsochrone(SphericallySymmetric):
    """
    Henon's isochrone model.

    """

    @property
    def name(cls):
        """
        The class name.

        """
        return "HenonIsochrone"

    @property
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
        Radial coordinate in units of $b$, the characteristic lengthscale
        within which the mass density is roughly constant.

        Returns
        -------

        out : ndarray
        Mass density in units of $M/b^3$.

        """
        a = np.sqrt(1. + r * r)
        return (3. * (1. + a) * a * a - r * r *
                (1. + 3. * a)) / (a * (a + 1.))**3. / (4. * np.pi)

    def deriv_mass_density(self, r):
        """
        The radial derivative of the profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $b$, the characteristic lengthscale
        within which the mass density is roughly constant.

        Returns
        -------

        out : ndarray
        Radial derivative of the mass density in units of $M/b^4$.

        """
        a = np.sqrt(1. + r * r)
        return -(r * (20. * (1. + a) + r * r *
                      (17. + 8. * a))) / (a * (1. + a))**4. / (4. * np.pi * a)

    def relative_potential(self, r, offset=0.):
        """
        The radial profile of the relative potential.
    
        Parameters
        ----------
        
        `r` : array_like
        Radial coordinate in units of $b$, the characteristic lengthscale
        within which the mass density is roughly constant.

        Returns
        -------

        out : ndarray
        Relative potential in units of $GM/b$.

        """
        return 1. / (1. + np.sqrt(1. + r * r))

    def deriv_relative_potential(self, r, offset=0.):
        """
        The radial derivative of the profile of the relative potential.
    
        Parameters
        ----------
        
        `r` : array_like
        Radial coordinate in units of $b$, the characteristic lengthscale
        within which the mass density is roughly constant.

        Returns
        -------

        out : ndarray
        Radial derivative of the relative potential in units of $GM/b^2$.

        """
        a = np.sqrt(1. + r * r)
        return -r / (a * (1. + a)**2.)

    def drho_dpsi(self, r, offset=0.):
        """
        The ratio `deriv_mass_density(r)` / `deriv_relative_potential(r)`.
    
        Parameters
        ----------
        
        `r` : array_like
        Radial coordinate in units of $b$, the characteristic lengthscale
        within which the mass density is roughly constant.

        Returns
        -------

        out : ndarray
        The ratio of derivatives in units of $1/Gb^2$.

        """
        a = np.sqrt(1. + r * r)
        return (2. * (1. - a + 4. * r**4.) + r * r *
                (10. + a)) / (4. * np.pi * r**2. * a**5.)

    def f(self, e):
        """
        The antiderivative of the DF.

        """
        print("WARNING: analytic Henon's isochrone DF not implemented yet.")
        return 1.

    def df(self, e):
        """
        The DF.

        """
        print("WARNING: analytic Henon's isochrone DF not implemented yet.")
        return 1.
