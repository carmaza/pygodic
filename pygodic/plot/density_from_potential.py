# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function :func:`density_from_potential`.

"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "cm"


def density_from_potential(spline, model, dense_npts=500, path=""):
    """
    Plot interpolation of the mass density and its derivative as a function of
    the relative potential.

    Parameters
    ----------

    spline : obj
        The B-spline representation of the curve. Must be a
        :class:`LogRhoVsLogPsi` object.

    model : object
        The model in consideration.

    dense_npts : int (optional, default: 500)
        The number of points used in the dense set of absisas.

    path : string (optional, default: the running folder)
        The path where to save the files.

    """
    knots = spline.logpsi[0], spline.logpsi[-1]
    logpsi_dense = np.log10(
        np.geomspace(np.power(10, knots[0]), np.power(10, knots[-1]),
                     dense_npts))

    # Plot log density vs log potential.
    plt.plot(spline.logpsi,
             spline.logrho,
             'o',
             color='pink',
             label="parametric")
    plt.plot(logpsi_dense,
             spline(logpsi_dense),
             color='red',
             label="interpolation",
             linewidth=2.5)

    plt.xlabel(r"$\log\Psi$", fontsize=20)
    plt.ylabel(r"$\log\rho$", fontsize=20, labelpad=18, rotation=0)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)

    filepath = f"{path}LogRhoVsLogPsi{model.name()}.pdf"
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print(f"File {filepath} saved.")

    # Plot dlogrho/dlogpsi vs log potential.
    plt.plot(spline.logpsi,
             spline.dlr_dlp,
             'o',
             color='pink',
             label="parametric")
    plt.plot(logpsi_dense,
             spline(logpsi_dense, nu=1),
             color='red',
             label="interpolation",
             linewidth=2.5)

    plt.xlabel(r"$\log\Psi$", fontsize=20)
    plt.ylabel(r"$\frac{d\log\rho}{d\log\Psi}$",
               fontsize=28,
               labelpad=34,
               rotation=0)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)

    filepath = f"{path}dLogRhodLogPsi{model.name()}.pdf"
    plt.savefig(filepath, bbox_inches='tight')
    plt.clf()
    print(f"File {filepath} saved.")
