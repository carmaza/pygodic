# Distributed under the MIT License.
# See LICENSE for details.

import abc


class SphericallySymmetric(metaclass=abc.ABCMeta):
    """
    Base class for self-gravitating spherically symmetric models.

    """

    @property
    @abc.abstractmethod
    def name():
        pass

    @property
    @abc.abstractmethod
    def has_analytic_df():
        pass

    @staticmethod
    @abc.abstractmethod
    def mass_density(r):
        pass

    @staticmethod
    @abc.abstractmethod
    def deriv_mass_density(r):
        pass

    @staticmethod
    @abc.abstractmethod
    def relative_potential(r, offset):
        pass

    @staticmethod
    @abc.abstractmethod
    def deriv_relative_potential(r, offset):
        pass

    @staticmethod
    @abc.abstractmethod
    def drho_dpsi(r, offset):
        pass
