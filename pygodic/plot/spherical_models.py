# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following functions:

- `plot.radial_profiles`
- `plot.density_vs_potential`
- `plot.available_spherical_profiles`.

"""

import matplotlib.pyplot as plt
import numpy as np

import pygodic.models as models

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def radial_profiles(models, r, normalized=False, path=""):
    """
    Plot radial profiles of the mass density and the relative potential for
    the given spherical models.

    Parameters
    ----------

    `models` : list
    List of `SphericallySymmetric` objects from which to extract the profiles.
    
    `r` : array_like
    Radial coordinate. It should be normalized to some characteristic radius.

    `normalized` : bool (optional, default: `False`)
    Whether to plot profiles normalized to their value at $r = 0$.

    `path` : string (optional, default: the running folder)
    The path where to save the plots.

    """

    def plot_profile(fieldname, ylabel, xlogscale=False, ylogscale=False):
        for model in models:
            radial_profile = None
            if fieldname == "MassDensity":
                radial_profile = model.mass_density
            elif fieldname == "RelativePotential":
                radial_profile = model.relative_potential

            profile = radial_profile(r)
            if normalized:
                profile = profile / radial_profile(0.)
            plt.plot(r, profile, '-', label=model.name)

        if xlogscale:
            plt.xscale('log')
        if ylogscale:
            plt.yscale('log')

        plt.xlabel("$r$", fontsize=20)
        plt.ylabel(ylabel, fontsize=20, labelpad=12, rotation=0)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=18)

        filepath = path + "{fieldname}.pdf".format(fieldname=fieldname)
        plt.savefig(filepath, bbox_inches='tight')
        plt.clf()
        print("File {path} saved.".format(path=filepath))

    plot_profile("MassDensity", r"$\rho$", xlogscale=True, ylogscale=True)
    plot_profile("RelativePotential", "$\Psi$", xlogscale=True)


def density_vs_potential(models, r, ylogscale=False, path=""):
    """
    Plot the mass density vs the relative potential parametrically for the
    given list of models, using the radial coordinate as parameter.

    Parameters
    ----------

    `models` : list
    List of `SphericallySymmetric` objects from which to extract the profiles.
    
    `r` : array_like
    Radial coordinate. It should be normalized to some characteristic radius.

    `ylogscale` : bool (optional, default: False)
    Whether to plot the vertical axis (mass density) in log scale.

    `path` : string (optional, default: the running folder)
    The path where to save the plots.

    """
    for model in models:
        plt.plot(model.relative_potential(r),
                 model.mass_density(r),
                 '.',
                 label=model.name)

    if ylogscale:
        plt.yscale("log")

    plt.xlabel("$\Psi$", fontsize=20)
    plt.ylabel(r"$\rho$", fontsize=20, rotation=0, labelpad=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.tight_layout()

    filepath = path + "DensityVsPotential.pdf"
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print("File {path} saved.".format(path=filepath))


def available_spherical_profiles():
    """
    Plot all available spherical profiles in a single figure.

    """
    available_models = [
        models.Exponential(),
        models.ExponentialLinear(),
        models.Jaffe(),
        models.HenonIsochrone(),
        models.Plummer()
    ]

    r = np.geomspace(1.e-6, 10., 1000)

    radial_profiles(available_models, r)
    density_vs_potential(available_models, r, ylogscale=True)
