# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np
import unittest

from pygodic import models
from pygodic.numalg import integrate

from .finite_difference import FirstDeriv


def _total_mass(model, cutoff=10.):
    """
    Helper function to compute the total mass by integrating the `model`'s
    spherically symmetric mass density.

    Notes
    -----

    Integral is split into the two regions

    (1) $r \in [0, c]$
    (2) $r \in [c, infinity)$

    where in region (2) we use the substitution $u = 1/r$ in order to reduce the
    integration domain to a finite range. The cutoff value $c$ is arbitrary, but
    choosing it around regions where the integrand is flatter should give more
    accurate results.

    """
    r = np.linspace(0., cutoff, 10000)
    mass_density = model.mass_density(r)
    mass_1 = 4. * np.pi * integrate.simpson(r * r * mass_density, r)

    u = np.linspace(0., 1. / cutoff, 10000)
    inv_u = 1. / (u + 1.e-12)  # Small offset to avoid division by zero.
    f_of_u = inv_u**4. * model.mass_density(inv_u)
    mass_2 = 4. * np.pi * integrate.simpson(f_of_u, u)

    return mass_1 + mass_2


class TestSphericalModels(unittest.TestCase):
    """
    Test classes derived from `SphericallySymmetric`.

    Current tests being performed:
    - Test that integrated mass density gives total mass = 1
    - Test that deriv_relative_potential and mass_density satisfy Poisson's equation.

    To test new models, simply add them to `models` in the end of `test` function.

    TO-DO: add test for deriv_mass_density, relative_potential, and drho_dpsi.
    TO-DO: add test for analytical DF's. Good candidate: integral(DF) = rho.

    """

    def test(self):
        # Test that mass density is normalized so that total mass = 1.
        def test_mass(model):
            mass = _total_mass(model)
            msg = f"mass = {mass} is not unity for {model.name} model."
            self.assertAlmostEqual(mass, 1., msg=msg)

        # Test that potential and density are related via Poisson's equation.
        def test_poisson_solution(model):
            deriv = FirstDeriv()

            # Dummy domain to check algebra. Must be dense enough for FDs.
            r = np.linspace(0.3, 0.7, 10000)

            deriv_potential = -model.deriv_relative_potential(r)
            lap_potential = deriv.second_order(r * r * deriv_potential,
                                               r) / (r * r)
            four_pi_density = 4. * np.pi * model.mass_density(r)
            msg = f"Poisson's equation not satisfied for {model.name} model."
            self.assertTrue(np.allclose(
                (lap_potential - four_pi_density) / four_pi_density,
                0.,
                atol=1.e-7),
                            msg=msg)

        finite_mass_models = [
            models.Exponential(),
            models.ExponentialLinear(),
            models.HenonIsochrone(),
            models.Plummer()
        ]

        for model in finite_mass_models:
            test_mass(model)
            test_poisson_solution(model)

        # Jaffe's model's mass is divergent, so we only test Poisson's equation.
        test_poisson_solution(models.Jaffe())
