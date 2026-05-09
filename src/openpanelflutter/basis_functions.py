"""Displacement functions and hierarchical polynomials for Ritz Method.

This module is the central repository for basis functions. To implement a new
set of functions:
1. Add a new evaluation method in the Function class.
2. Update the __init__ logic to recognize the new type.
3. Add the corresponding constant string in 'definitions.py'.
"""

import numpy as np

from .definitions import BasisFunction


def dfat(n: int) -> int:
    """Compute the double factorial of n (n!!)."""
    n = int(n)
    if n == 0 or n == -1 or n == 1:
        return 1
    else:
        return n * dfat(n - 2)


def fat(n: int) -> int:
    """Compute the factorial of n (n!)."""
    n = int(n)
    if n == 1 or n == 0 or n == -1:
        return 1
    else:
        return n * fat(n - 1)


class Function:
    """Representation of a 1D displacement function and its derivatives.

    Provides a unified interface for evaluating different mathematical bases
    used in the Ritz method for structural analysis.
    """

    def __init__(self, r: int, x: np.ndarray, f_type: str):
        """Initialize and evaluate the function and its derivatives.

        Args:
            r (int): The index or order of the function.
            x (np.ndarray): Array of coordinates where function is evaluated.
            f_type (str): Type of basis (defined in definitions.py).
        """
        self.r = r
        self.x = x

        # Mapping the function type to specific evaluation methods
        if f_type == BasisFunction.BARDELL:
            self._compute_bardell()
        elif f_type == BasisFunction.SINES:
            self._compute_sines()
        elif f_type == BasisFunction.COSINES:
            self._compute_cosines()
        else:
            # Error for unsupported function type
            raise ValueError(
                f"Unknown function type: '{f_type}'. "
                f"Please check definitions.py for supported types."
            )

    def _compute_bardell(self):
        """Evaluate Bardell's hierarchical polynomials and derivatives.

        Ref: Bardell, N. S. (1992). The free vibration of skewed plates.
        """
        r = self.r

        r_int, r_dec = divmod(r, 2)
        if r == 1:
            # 1/2-3/4*x+1/4*(x**3)
            self.func = np.array([[1 / 4, -3 / 4, 1 / 2], [3, 1, 0]])
        if r == 2:
            # 1/8-1/8*x-1/8*(x**2)+1/8*(x**3)
            self.func = np.array(
                [[1 / 8, -1 / 8, -1 / 8, 1 / 8], [3, 2, 1, 0]]
            )
        if r == 3:
            # 1/2+3/4*x-1/4*(x**3)
            self.func = np.array([[-1 / 4, 3 / 4, 1 / 2], [3, 1, 0]])
        if r == 4:
            # -1/8-1/8*x+1/8*(x**2)+1/8*(x**3)
            self.func = np.array(
                [[1 / 8, 1 / 8, -1 / 8, -1 / 8], [3, 2, 1, 0]]
            )
        if r > 4:
            self.func = np.zeros((2, r_int + 1))
            for n in range(0, r_int + 1):
                self.func[0, n] = ((-1) ** n) * dfat(2 * r - 2 * n - 7)
                self.func[0, n] = self.func[0, n] / (
                    (2**n) * fat(n) * fat(r - 2 * n - 1)
                )
                self.func[1, n] = r - 2 * n - 1
        if self.func[1, -1] == -1:
            self.func = np.delete(self.func, -1, -1)

        self._derive_polynomials()
        self._evaluate_polynomials()

    def _derive_polynomials(self):
        """Generate coefficients for 1st, 2nd, and 3rd derivatives."""

        def get_der(poly):
            der = np.zeros((2, poly.shape[1]))
            for col in range(poly.shape[1]):
                der[0, col] = poly[0, col] * poly[1, col]
                der[1, col] = poly[1, col] - 1
            # Clean up zero-power terms (exponent = -1)
            return np.delete(der, -1, -1) if der[1, -1] == -1 else der

        self.dfunc = get_der(self.func)
        self.d2func = get_der(self.dfunc)
        self.d3func = get_der(self.d2func)

    def _evaluate_polynomials(self):
        """Numerically evaluate polynomial values at coordinates x."""

        def poly_val(coeffs, x):
            val = 0
            for col in range(coeffs.shape[1]):
                val += (x ** coeffs[1, col]) * coeffs[0, col]
            return val

        self.vfunc = poly_val(self.func, self.x)
        self.vdfunc = poly_val(self.dfunc, self.x)
        self.vd2func = poly_val(self.d2func, self.x)
        self.vd3func = poly_val(self.d3func, self.x)

    def _compute_sines(self):
        """Evaluate sine basis functions and derivatives."""
        arg = self.r * np.pi * (self.x + 1) / 2
        factor = self.r * np.pi / 2

        self.vfunc = np.sin(arg)
        self.vdfunc = np.cos(arg) * factor
        self.vd2func = -np.sin(arg) * (factor**2)
        self.vd3func = -np.cos(arg) * (factor**3)

    def _compute_cosines(self):
        """Evaluate cosine basis functions and derivatives."""
        r, x = self.r, self.x
        arg = r * np.pi * (x + 1) / 2
        factor = r * np.pi / 2

        self.vfunc = np.cos(arg)
        self.vdfunc = -np.sin(arg) * factor
        self.vd2func = -np.cos(arg) * (factor**2)
        self.vd3func = np.sin(arg) * (factor**3)

    @staticmethod
    def evaluate_basis_product(
        base_set: list,
        xys: np.ndarray,
        f_type: str,
        attr_xi: str,
        attr_eta: str,
    ) -> np.ndarray:
        """Evaluates products of shape function derivatives for a given set.

        This centralizes the assembly of N matrices for Panels and Stiffeners.
        """
        xi_pts, eta_pts = xys[:, 0], xys[:, 1]
        results = []

        for r in base_set:
            f_xi = Function(r[0], xi_pts, f_type)
            f_eta = Function(r[1], eta_pts, f_type)

            val_xi = getattr(f_xi, attr_xi)
            val_eta = getattr(f_eta, attr_eta)

            results.append([val_xi * val_eta])

        return np.array(results).T
