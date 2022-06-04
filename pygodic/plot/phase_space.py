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


def speed_moment_profile(r, moment, nlabel, model, path=""):
    """
    For the given model, plot the n-th speed moment as a function of the
    radial coordinate.

    Parameters
    ----------

    `r` : array_like
    The set of radial coordinates.

    `moment` : array_like
    The speed moment evaluated at `r`.

    `nlabel` : string
    The value of $n$ (to be used in the name of the file).

    `model` : object
    The model in consideration.

    `path` : string (optional, default: the running folder)
    The path where to save the plot.

    """
    labels = {
        "0": "0th moment (density)",
        "1": "1st moment (mean speed)",
        "2": "2nd moment (mean speed squared)"
    }

    label = ""
    if nlabel in labels.keys():
        label = labels[nlabel]
    else:
        label = "{n}th moment".format(n=nlabel)
    plt.plot(r, moment, label=label, color="navy")

    if nlabel == "0":
        plt.plot(r, model.mass_density(r), color="green", label="analytic")
        plt.yscale("log")
        plt.gca().set_aspect(0.8)

    plt.xlabel("$r/R$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=12)
    plt.grid(color="grey")

    filepath = path + "{n}thSpeedMomentVsR{model}.pdf".format(
        n=nlabel, model=model.name())
    plt.savefig(filepath, bbox_inches="tight")
    plt.clf()
    print("File {path} saved.".format(path=filepath))


def dispersion_profile(r, v_mean, v_sqrd_mean, model, path=""):
    """
    For the given model, plot radial profile of the speed dispersion.

    Parameters
    ----------

    `r` : array_like
    The set of radial coordinates.

    `v_mean, v_sqrd_mean` : array_like, array_like
    The mean speed and the mean speed squared evaluated at `r`.

    `model` : object
    The model in consideration.

    `path` : string (optional, default: the running folder)
    The path where to save the plot.

    """
    plt.plot(r, v_sqrd_mean, color="navy", label="$\overline{v^2}$")
    plt.plot(r, v_mean * v_mean, color="green", label="$\overline{v}^2$")
    plt.plot(r,
             v_sqrd_mean - v_mean * v_mean,
             color="yellowgreen",
             label=r"$\sigma^2$")

    plt.xlabel("$r/R$", fontsize=20)
    plt.yscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.gca().set_aspect(2.5)
    plt.grid(color="grey")

    filepath = path + "Dispersion{model}.pdf".format(model=model.name())
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print("File {path} saved.".format(path=filepath))
