# Distributed under the MIT License.
# See LICENSE for details.

import abc


class SphericallySymmetric(metaclass=abc.ABCMeta):
    """
    Base class for self-gravitating spherically symmetric models.

    """

    @property
    def name():
        raise NotImplementedError()

    @property
    def has_analytic_df():
        raise NotImplementedError()

    @staticmethod
    def mass_density(r):
        raise NotImplementedError()

    @staticmethod
    def deriv_mass_density(r):
        raise NotImplementedError()

    @staticmethod
    def relative_potential(r, offset):
        raise NotImplementedError()

    @staticmethod
    def deriv_relative_potential(r, offset):
        raise NotImplementedError()

    @staticmethod
    def drho_dpsi(r, offset):
        raise NotImplementedError()
