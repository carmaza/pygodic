# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the base class for spherical models.

"""

import abc


class SphericallySymmetric(metaclass=abc.ABCMeta):
    """
    Base class for self-gravitating spherically symmetric models.

    """

    @classmethod
    @abc.abstractmethod
    def name(cls):
        """
        The class name.

        """

    @classmethod
    @abc.abstractmethod
    def has_analytic_df(cls):
        """
        Whether the model has an analytic DF.

        """

    @abc.abstractmethod
    def mass_density(self, r):
        """
        Return the mass density as a function of the radial coordinate.

        """

    @abc.abstractmethod
    def deriv_mass_density(self, r):
        """
        Return the derivative of the mass density with respect to the radial
        coordinate.

        """

    @abc.abstractmethod
    def relative_potential(self, r, offset):
        """
        Return the relative potential as a function of the radial coordinate.

        """

    @abc.abstractmethod
    def deriv_relative_potential(self, r, offset):
        """
        Return the derivative of the relative potential with respect to the
        radial coordinate.

        """

    @abc.abstractmethod
    def drho_dpsi(self, r, offset):
        """
        Return the derivative of the relative potential with respect to the
        mass density.

        """
