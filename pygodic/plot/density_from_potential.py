# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function `plot.density_spline`.

"""

import matplotlib.pyplot as plt
import numpy as np

from pygodic.numalg import interpolate

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def density_spline(psi,
                   rho,
                   drho_dpsi,
                   rho_spline,
                   drho_dpsi_spline,
                   model,
                   npts=500,
                   path=""):
    """
    Plot interpolation of the mass density and its derivative as a function of
    the relative potential.

    Parameters
    ----------

    `psi`, `rho`, drho_dpsi` : array_like, array_like, array_like
    The points used to get the interpolation.

    `rho_spline`, `drho_dpsi_spline` : object, object
    The B-spline representations of the interpolant of the mass density and
    its derivative.

    `model` : object
    The model in consideration.

    `npts` : int (optional, default: 500)
    The number of points used in the dense set of absisas.

    `path` : string (optional, default: the running folder)
    The path where to save the files.

    """
    psi_dense = np.geomspace(psi[0], psi[-1], npts)

    # Plot density vs potential.
    plt.plot(psi, rho, 'o', color='pink', label="parametric")
    plt.plot(psi_dense,
             interpolate.spline_evaluation(psi_dense, rho_spline, der=0),
             color='red',
             label="interpolation",
             linewidth=2.5)

    plt.xlabel("$\Psi$", fontsize=20)
    plt.ylabel(r"$\rho$", fontsize=20, labelpad=18, rotation=0)
    plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    filepath = path + "RhoVsPsi{model}.pdf".format(model=model.name)
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print("File {path} saved.".format(path=filepath))

    # Plot drho/dpsi vs potential.
    plt.plot(psi, drho_dpsi, 'o', color='pink', label="parametric")
    plt.plot(psi_dense,
             interpolate.spline_evaluation(psi_dense, drho_dpsi_spline, der=0),
             color='red',
             label="direct interpolation",
             linewidth=2.5)
    plt.plot(psi_dense,
             interpolate.spline_evaluation(psi_dense, rho_spline, der=1),
             color='navy',
             label="deriv interpolation",
             linewidth=1.2)

    plt.xlabel("$\Psi$", fontsize=20)
    plt.ylabel(r"$\frac{d\rho}{d\Psi}$", fontsize=30, labelpad=24, rotation=0)
    plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)

    filepath = path + "dRhodPsi{model}.pdf".format(model=model.name)
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print("File {path} saved.".format(path=filepath))
