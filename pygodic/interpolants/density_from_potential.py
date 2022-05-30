# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np

import pygodic.numalg.interpolate as interpolate
import pygodic.plot as plot


def density_from_potential(model,
                           xi_min,
                           xi_max,
                           n_pts,
                           k=3,
                           make_plots=False):
    """
    For the given model of known radial profiles of the mass density $\rho(r)$
    and the relative potential $\Psi(r)$, get interpolants for the mass density
    and its derivative as a function of the relative potential.

    Details
    -------

    The interpolant will be a B-spline of the given order `k`. The absisas of
    the interpolation will be the set $\{r_j\} = \{\exp(\\xi_j)\}$,
    $j = 0,1,\hdots$, where $\\xi_j$ is a uniformly distributed parameter in the
    range $[\\xi_\text{min}, \\xi_\text{max}]$. The ordinates of the
    interpolation for the mass density and its derivative will be, respectively,
    the sets $\{\rho(r_j)\}$ and $\{u(\r_j)\}$, where
    $u(r) \equiv \rho'(r)/\Psi'(r)$.

    Parameters
    ----------

    `model` : object
    The model in consideration. Must be a `SphericallySymmetric` object.

    `xi_min`, `xi_max` : float, float
    The minimum and maximum values for the parameter $\\xi$.

    `n_pts` : int
    The number of points used to get the interpolants.

    `k` : int (optional, default: 3)
    The order of the interpolations. Must be in the range [1, 5].

    `make_plots` : bool (optional, default: False)
    Whether to make plots of the interpolants evaluated at a dense set, along
    with the points used to get the interpolants.

    Returns
    -------

    out : ndarray, object, object
    Tuple containing, respectively, the set of potentials used to obtain the
    interpolants, the B-spline representation of the interpolant of the mass
    density, and the B-spline representation of the interpolant of the
    derivative of the mass density.

    Notes
    -----

    - The radial profiles are assummed to be decreasing functions of the radial
      coordinate.

    """
    radius = np.exp(np.linspace(xi_min, xi_max, n_pts))

    # Radial profiles are decreasing so we need to flip arrays to interpolate.
    psi = np.flip(model.relative_potential(radius))
    rho = np.flip(model.mass_density(radius))
    drho_dpsi = np.flip(model.drho_dpsi(radius))

    rho_spline = interpolate.spline_representation(psi, rho, k=k)
    drho_dpsi_spline = interpolate.spline_representation(psi, drho_dpsi, k=k)

    if make_plots:
        plot.density_spline(psi, rho, drho_dpsi, rho_spline, drho_dpsi_spline,
                            model)

    return psi, rho_spline, drho_dpsi_spline
