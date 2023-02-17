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

    @property
    @abc.abstractmethod
    def name():
        """
        The class name.

        """

    @property
    @abc.abstractmethod
    def has_analytic_df():
        """
        Whether the model has an analytic DF.

        """

    @staticmethod
    @abc.abstractmethod
    def mass_density(r):
        """
        Return the mass density as a function of the radial coordinate.

        """

    @staticmethod
    @abc.abstractmethod
    def deriv_mass_density(r):
        """
        Return the derivative of the mass density with respect to the radial
        coordinate.

        """

    @staticmethod
    @abc.abstractmethod
    def relative_potential(r, offset):
        """
        Return the relative potential as a function of the radial coordinate.

        """

    @staticmethod
    @abc.abstractmethod
    def deriv_relative_potential(r, offset):
        """
        Return the derivative of the relative potential with respect to the
        radial coordinate.

        """

    @staticmethod
    @abc.abstractmethod
    def drho_dpsi(r, offset):
        """
        Return the derivative of the relative potential with respect to the
        mass density.

        """
