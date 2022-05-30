# Distributed under the MIT License.
# See LICENSE for details.

from abc import ABCMeta, abstractmethod, abstractproperty


class SphericallySymmetric():
    """
    Base class for self-gravitating spherically symmetric models.

    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def name():
        pass

    @abstractproperty
    def has_analytic_df():
        pass

    @abstractmethod
    def mass_density(r):
        pass

    @abstractmethod
    def deriv_mass_density(r):
        pass

    @abstractmethod
    def relative_potential(r, offset):
        pass

    @abstractmethod
    def deriv_relative_potential(r, offset):
        pass

    @abstractmethod
    def drho_dpsi(r, offset):
        pass
