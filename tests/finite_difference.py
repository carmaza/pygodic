# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np


class FirstDeriv:

    @staticmethod
    def second_order(f, x):
        """
        Approximate the first derivative using second-order finite difference
        on a grid of $N$ arbitrarily spaced points labeled $j = 0, \hdots, N-1$.

        Lowest point uses the asymmetric stencil f[0], f[1], f[2].

        Uppest point uses the asymmetric stencil f[N-3], f[N-2], f[N-1].

        Middle points use the symmetric stencil f[j-1], f[j], f[j+1].

        Parameters :
        ------------

        `f` : ndarray
        The values of the function to differentiate on the grid.

        `x` : ndarray
        The spatial grid. (Must be of the same size of `f`.)

        Returns :
        ---------

        out : ndarray
        The derivative of the function on each grid point.

        """
        n = f.size
        dfdx = np.zeros(n)

        delta_0 = x[1] - x[0]
        delta_1 = x[2] - x[1]

        dfdx[0] = (-(2. * delta_0 + delta_1) * f[0] / delta_0 -
                   delta_0 * f[2] / delta_1) / (delta_0 + delta_1) + (
                       delta_0 + delta_1) * f[1] / delta_0 / delta_1

        for j in range(1, n - 1):
            delta_j = x[j + 1] - x[j]
            delta_jm1 = x[j] - x[j - 1]

            dfdx[j] = (delta_jm1 * f[j + 1] / delta_j - delta_j * f[j - 1] /
                       delta_jm1) / (delta_j + delta_jm1) + (
                           delta_j - delta_jm1) * f[j] / delta_j / delta_jm1

        delta_nm2 = x[n - 1] - x[n - 2]
        delta_nm3 = x[n - 2] - x[n - 3]

        dfdx[n - 1] = (
            (2. * delta_nm2 + delta_nm3) * f[n - 1] / delta_nm2 +
            delta_nm2 * f[n - 3] / delta_nm3) / (delta_nm2 + delta_nm3) - (
                delta_nm2 + delta_nm3) * f[n - 2] / delta_nm2 / delta_nm3

        return dfdx
