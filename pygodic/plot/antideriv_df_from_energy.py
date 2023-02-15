# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function `plot.antideriv_df_from_energy`.

"""

import matplotlib.pyplot as plt
import numpy as np

from pygodic.numalg import interpolate

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def antideriv_df_from_energy(energy,
                             antideriv_df,
                             antideriv_df_spline,
                             model,
                             npts=500,
                             path=""):
    """
    Plot interpolation of the DF and its antiderivative as a function of the
    relative energy.

    Parameters
    ----------

    `energy`, `antideriv_df` : array_like, array_like
    The points used to get the interpolation.

    `antideriv_df_spline` : object
    The B-spline representation of the interpolant of the antiderivative of the
    DF.

    `model` : object
    The model in consideration.

    `npts` : int (optional, default: 500)
    The number of points used in the dense set of absisas.

    `path` : string (optional, default: the running folder)
    The path where to save the files.

    """
    energy_dense = np.geomspace(energy[0], energy[-1], npts)

    # Plot antiderivative of DF.
    plt.plot(energy, antideriv_df, 'o', color='pink', label="GJ quadrature")
    plt.plot(energy_dense,
             interpolate.spline_evaluation(energy_dense,
                                           antideriv_df_spline,
                                           der=0),
             color='red',
             label="interpolation",
             linewidth=2.5)

    if model.has_analytic_df:
        plt.plot(energy_dense,
                 model.antideriv_df(energy_dense),
                 '--',
                 color='purple',
                 label="analytic",
                 linewidth=1.8)

    plt.xlabel(r"$\mathcal{E}$", fontsize=20)
    plt.ylabel(r"$F(\mathcal{E})$", fontsize=20, labelpad=22, rotation=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)

    filepath = f"{path}FvsE{model.name}.pdf"
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print(f"File {filepath} saved.")

    # Plot DF.
    plt.plot(energy,
             interpolate.spline_evaluation(energy, antideriv_df_spline, der=1),
             'o',
             color='pink',
             label="GJ quadrature")
    plt.plot(energy_dense,
             interpolate.spline_evaluation(energy_dense,
                                           antideriv_df_spline,
                                           der=1),
             color='red',
             label="interpolation",
             linewidth=2.5)

    if model.has_analytic_df:
        plt.plot(energy_dense,
                 model.df(energy_dense),
                 '--',
                 color='purple',
                 label="analytic",
                 linewidth=1.8)

    plt.xlabel(r"$\mathcal{E}$", fontsize=20)
    plt.ylabel(r"$f(\mathcal{E})$", fontsize=20, labelpad=22, rotation=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)

    filepath = f"{path}DFvsE{model.name}.pdf"
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print(f"File {filepath} saved.")
