# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function `plot.df_local_errors`.

"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pygodic import eddington_inversion, interpolants

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def df_local_errors(model,
                    xi_min,
                    xi_max,
                    pts_ene,
                    n_quad_max=10,
                    horizontal=False,
                    relative=True,
                    path=""):
    """
    Plot sequence of local errors in the antiderivative of the DF.

    Parameters
    ----------

    `model` : object
    The model in consideration. Assumed to be a `SphericallySymmetric` object.

    `xi_min`, `xi_max` : float, float
    The bounds in the parametric representation of the radial coordinate used
    to obtain the interpolant for the mass density as a function of the
    relative potential. The interpolant is needed to evaluate the
    antiderivative of the DF numerically.

    `pts_ene` : int
    The number of points in the relative energy domain.

    `n_quad_max` : int (optional, default: 10)
    The maximum number of nodes in the sequence.

    `horizontal` : bool (optional, default: False)
    Whether to show the sequence of panels horizontally instead.

    `relative` : bool (optional, default: True)
    Whether to plot error normalized by analytic value instead.

    `path` : string (optional, default: the running folder)
    The path where to save the plot.

    """
    radial_resolutions = [100, 1000, 10000]

    fig, axs = plt.subplots(len(radial_resolutions),
                            1,
                            figsize=(5, 9),
                            sharex=True)
    xforylabel, yforxlabel = -0.1, 0.03
    fig.subplots_adjust(hspace=0.1)

    if horizontal:
        fig, axs = plt.subplots(1,
                                len(radial_resolutions),
                                figsize=(12, 3.5),
                                sharey=True)
        xforylabel, yforxlabel = 0.02, -0.1
        fig.subplots_adjust(wspace=0.1)

    for j, axis in enumerate(axs):
        potential, _, drho_dpsi_spline = interpolants.density_from_potential(
            model, xi_min, xi_max, radial_resolutions[j], k=3)

        energy = np.geomspace(potential[0], potential[-1], pts_ene)
        antideriv_df = model.antideriv_df(energy)

        for n in range(1, n_quad_max + 1):
            f_n = eddington_inversion.antideriv_df(energy, energy[0],
                                                   drho_dpsi_spline, n)
            error = np.abs((f_n - antideriv_df))
            if relative:
                error = error / antideriv_df

            axis.plot(energy, error, label=rf"$n={n}$")
            axis.grid(visible=True, alpha=0.5)
            if horizontal:
                axis.set_title(rf"radial points: $10^{str(j + 2)}$",
                               fontsize=22)

            axis.set_xscale("log")
            axis.set_yscale("log")

            fontsize = 22 if horizontal else 18
            plt.setp(axis.get_xticklabels(), fontsize=fontsize)
            plt.setp(axis.get_yticklabels(), fontsize=fontsize)

    fig.supxlabel(r"$\mathcal{E}$", fontsize=24, y=yforxlabel)
    fig.supylabel(r"$|F_\mathrm{num} - F_\mathrm{ana}| / F_\mathrm{ana}$",
                  fontsize=24,
                  x=xforylabel)

    plt.legend(borderpad=0.2,
               columnspacing=1.,
               fontsize=16,
               handlelength=0.5,
               handletextpad=0.4,
               labelspacing=0.1,
               loc='best',
               ncol=2)

    filepath = f"{path}ErrorFvsE{model.name()}.pdf"
    plt.savefig(filepath, bbox_inches='tight')

    # Ignore unnecessary warning originated from known bug:
    # https://github.com/matplotlib/matplotlib/issues/9970
    warnings.filterwarnings(
        "ignore",
        message=
        "Attempt to set non-positive ylim on a log-scaled axis will be ignored."
    )

    plt.clf()
    print(f"File {filepath} saved.")

    # Manually set figsize back to default value.
    plt.figure(figsize=(6.4, 4.8))
