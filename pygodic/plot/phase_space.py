# Distributed under the MIT License.
# See LICENSE for details.

import matplotlib.pyplot as plt
import numpy as np

import pygodic.eddington_inversion as eddington_inversion
import pygodic.functions_of_phase_space as functions_of_phase_space

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def relative_energy_contours(radius, speed, model, path=""):
    """
    For the given model, plot contours of the relative energy as a function
    of the radial coordinate and the speed.

    Parameters
    ----------

    `radius`, `speed` : meshgrid, meshgrid
    The phase space coordinates.

    `model` : object
    The model in consideration.

    `path` : string (optional, default: the running folder)
    The path where to save the plot.

    """
    energy = functions_of_phase_space.relative_energy(radius, speed, model)

    plt.contour(radius, speed, energy, 15)
    zero_contour = plt.contour(radius,
                               speed,
                               energy, [0.],
                               alpha=0.9,
                               colors='white',
                               linewidths=2.5)
    plt.contourf(radius, speed, energy, 15, alpha=0.5)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel("$r$", fontsize=20)
    plt.ylabel("$v$", fontsize=20, labelpad=15, rotation=0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    filepath = path + "RelativeEnergyContours{model}.pdf".format(
        model=model.name())
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print("File {path} saved.".format(path=filepath))


def df_contours(radius,
                speed,
                energy_min,
                antideriv_df_spline,
                model,
                spherical=False,
                logscale=False,
                path=""):
    """
    For the given model, plot contours of the DF as a function of the radial
    coordinate and the speed.

    Parameters
    ----------

    `radius`, `speed` : meshgrid, meshgrid
    The phase space coordinates.

    `energy_min` : float
    The relative energy below which the DF vanishes. Should be the minimum
    energy for which the interpolant of the antiderivative of the DF is valid.

    `antideriv_df_spline` : object
    The B-spline representation of the interpolant of the DF's antiderivative.

    `model` : object
    The model in consideration.

    `spherical` : bool (optional, default: False)
    Whether to plot spherical DF instead. (The spherical DF equals the
    Cartesian DF times $4\pi v^2$.)

    `logscale` : bool (optional, default: False)
    Whether to plot the contours of log10(DF) instead.

    `path` : string (optional, default: the running folder)
    The path where to save the plot.

    """
    energy = functions_of_phase_space.relative_energy(radius, speed, model)

    df = eddington_inversion.df(energy, energy_min, antideriv_df_spline)
    if spherical:
        df = 4. * np.pi * speed**2. * df

    contours = None
    if logscale:
        # Hack to leave plot empty wherever log10(DF) is not defined.
        df = np.where(df > 0., df, np.nan)
        log_df = np.log10(df)
        contours = plt.contour(radius, speed, log_df, 20)
        plt.contourf(radius, speed, log_df, 20, alpha=0.3)

    else:
        plt.contour(radius, speed, df, 20)
        # Plot specific contour where DF = 0. However, since the plot contains
        # not a single contour but a whole region where DF = 0, we must use a
        # contour level *close* but not equal to zero.
        contours = plt.contour(radius,
                               speed,
                               df, [1.e-9],
                               alpha=0.9,
                               colors='white')
        plt.contourf(radius, speed, df, 20, alpha=0.5)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel("$r$", fontsize=20)
    plt.ylabel("$v$", fontsize=20, labelpad=15, rotation=0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    filepath = path + "DFContours{model}.pdf".format(model=model.name())
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print("File {path} saved.".format(path=filepath))
