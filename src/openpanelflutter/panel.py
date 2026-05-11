"""Module for the Ritz-based Panel structural model.

This module defines the Panel class, which handles geometry, laminate
integration, and the setup of kinematic basis functions for panel analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from matplotlib import cm

from .basis_functions import Function
from .boundary import get_bardell_indices
from .definitions import (
    BasisFunction,
    BoundaryCondition,
    StructuralTheory,
    apply_plot_style,
)
from .material import Laminate
from .stiffener import Stiffener

apply_plot_style()


class Panel:
    """Composite panel model using the Ritz Method.

    Coordinates the geometric, material, and kinematic properties required
    to assemble the global stiffness and mass matrices.
    """

    def __init__(
        self,
        length: float,
        width: float,
        laminate: Laminate,
        radius: float = 0.0,
        beta: float = 0.0,
    ):
        """Initialize the panel with geometry and material properties.

        Args:
            length (float): Length along the x-axis (a) [m].
            width (float): Width along the y-axis (b) [m].
            laminate (Laminate): Laminate object containing material data.
            radius (float, optional): Curvature radius [m]. Defaults to 0.0.
            beta (float, optional): Skew angle in degrees. Defaults to 0.0.
        """
        self.a = length
        self.b = width
        self.radius = radius
        self.beta = np.radians(beta)

        # Dependency Injection of the Laminate
        self.laminate = laminate
        self.laminate.compute_ABD()

        # Support for elastic boundary conditions
        self.torsional_springs = {
            "left": 0.0,  # xi = -1
            "right": 0.0,  # xi = +1
            "bottom": 0.0,  # eta = -1
            "top": 0.0,  # eta = +1
        }

        # Support for linear spring at specific points
        self.point_springs = []

        # Support for stiffeners
        self.stiffeners = []

    def setup_kinematics(
        self,
        n_func: int,
        theory: StructuralTheory = StructuralTheory.REISSNER_MINDLIN,
        basis_type: BasisFunction = BasisFunction.BARDELL,
        n_gauss: int = -1,
        boundary_conditions: BoundaryCondition = BoundaryCondition.SSSS,
        **kwargs,
    ):
        """Configure the Ritz basis functions and structural theory.

        Args:
            n_func (int): Number of functions in each direction for all DOFs.
            theory (StructuralTheory): RM (FSDT) or KF (Kirchhoff).
            basis_type (BasisFunction): Bardell, Sines, or Cosines.
            n_gauss (int, optional): Number of Gauss points. Defaults to -1.
            boundary_conditions (str, optional): BC string (e.g., 'CFFF').
            **kwargs: Additional parameters like 'nlin' for integration order.

        Raises:
            ValueError: If integration points are not provided for
            non Bardell functions.
        """
        self.n_func = n_func
        self.theory = theory
        self.basis_type = basis_type
        self.n_gauss = n_gauss
        self.boundary = boundary_conditions

        # Safety validation for integration
        if n_gauss == -1 and basis_type != BasisFunction.BARDELL:
            raise ValueError(
                "Numerical integration points (n_gauss) must be explicitly "
                "provided for this basis functions."
            )

        # Retrieve indices for the displacement fields based on BCs
        if self.basis_type == BasisFunction.BARDELL:
            # BC function returns lists of indices for each DOF
            (m_u, n_u, m_v, n_v, m_w, n_w, m_bx, n_bx, m_by, n_by) = (
                get_bardell_indices(theory, boundary_conditions, n_func)
            )
        else:
            # For trigonometric series, indices are typically sequential
            m_u = n_u = m_v = n_v = m_w = n_w = list(range(1, n_func + 1))

            # Rotation DOFs only exist in Reissner-Mindlin
            if self.theory == StructuralTheory.REISSNER_MINDLIN:
                m_bx = n_bx = m_by = n_by = list(range(1, n_func + 1))
            else:
                m_bx = n_bx = m_by = n_by = []

        # Generate 2D basis combinations
        self.base_u = [[i, j] for i in m_u for j in n_u]
        self.base_v = [[i, j] for i in m_v for j in n_v]
        self.base_w = [[i, j] for i in m_w for j in n_w]
        self.base_bx = [[i, j] for i in m_bx for j in n_bx]
        self.base_by = [[i, j] for i in m_by for j in n_by]

        # Initialize quadrature and mesh
        self._initialize_quadrature(**kwargs)

    def _initialize_quadrature(self, **kwargs):
        """Prepare Gauss-Legendre points, weights, and integration mesh.

        This method computes the Jacobian and refreshes stiffener positions
        if the mesh is updated.
        """
        # Determine minimum required Gauss points for Bardell polynomials
        if self.basis_type == BasisFunction.BARDELL:
            # Check for non-linear analysis flag
            nlin = kwargs.get("nlin", 0)
            # Standard rule for exact integration of polynomials
            min_gauss = int(
                np.ceil(((2 + 2 * nlin) * (self.n_func + 1) + 1) * 0.5)
            )

            if self.n_gauss < min_gauss:
                self.n_gauss = min_gauss

        # Generate 1D Gauss-Legendre quadrature
        xi_g, wi_g = np.polynomial.legendre.leggauss(self.n_gauss)
        self.xi = xi_g
        self.wi = wi_g

        # Build 2D mesh (xi, eta, i_xi, j_eta, weight_prod)
        # Using meshgrid for vectorized coordinate generation
        xi_coords, eta_coords = np.meshgrid(xi_g, xi_g, indexing="ij")
        wi_x, wi_y = np.meshgrid(wi_g, wi_g, indexing="ij")

        # indices for mapping
        ii, jj = np.meshgrid(range(len(xi_g)), range(len(xi_g)), indexing="ij")

        self.mesh = np.column_stack(
            (
                xi_coords.ravel(),
                eta_coords.ravel(),
                ii.ravel(),
                jj.ravel(),
                (wi_x * wi_y).ravel(),
            )
        )

        # Calculate Geometric Jacobian for the skewed/rectangular panel
        # J = (a/2) * (b/2) * cos(beta)
        self.jac = self.a * self.b * np.cos(self.beta) / 4.0

        self._eval_matrices()

        # Update stiffeners if they exist
        if hasattr(self, "stiffeners") and self.stiffeners:
            existing_stiffs = self.stiffeners[:]
            self.stiffeners = []  # Reset and re-insert
            for stf in existing_stiffs:
                self.insert_stiffener(
                    stf.x0,
                    stf.y0,
                    stf.x1,
                    stf.y1,
                    stf.height,
                    stf.laminate,
                    stf.gap,
                    stf.side,
                )
                # self.stiffeners[-1].init_nlin(self)
        else:
            self.stiffeners = []

    def _eval_matrices(self):
        """Assemble global stiffness (K), mass (M), and aerodynamic matrices.

        This method computes the laminate inertia properties, evaluates shape
        function products across the quadrature mesh, applies skew
        transformations,and performs numerical integration using tensordot
        for efficiency.
        """
        # 1. Compute Laminate Inertia Terms (I1, I2, I3)
        # ---------------------------------------------------------------------
        i1, i2, i3 = 0.0, 0.0, 0.0
        z_bot = -self.laminate.total_thickness / 2.0

        for ply in self.laminate.plies:
            rho = ply.material.rho
            z_top = z_bot + ply.h

            # Analytical integration of rho * z^n over the ply thickness
            i1 += rho * (z_top - z_bot)
            i2 += rho * (z_top**2 - z_bot**2) / 2.0
            i3 += rho * (z_top**3 - z_bot**3) / 3.0
            z_bot = z_top

        # 2. Define Constitutive and Inertia Matrices (C0 and I0)
        # ---------------------------------------------------------------------
        if self.theory == StructuralTheory.REISSNER_MINDLIN:
            # RM: [A B 0; B D 0; 0 0 As]
            self.c0 = np.block(
                [
                    [self.laminate.A, self.laminate.B, np.zeros((3, 2))],
                    [self.laminate.B, self.laminate.D, np.zeros((3, 2))],
                    [np.zeros((2, 6)), self.laminate.A_s],
                ]
            )
            # Mass inertia matrix for RM theory
            self.i0 = np.array(
                [
                    [i1, 0, 0, i2, 0],
                    [0, i1, 0, 0, i2],
                    [0, 0, i1, 0, 0],
                    [i2, 0, 0, i3, 0],
                    [0, i2, 0, 0, i3],
                ]
            )
        else:  # Kirchhoff theory
            self.c0 = np.block(
                [
                    [self.laminate.A, self.laminate.B],
                    [self.laminate.B, self.laminate.D],
                ]
            )
            # Mass inertia for KF (negative i2 due to w derivatives)
            self.i0 = np.array(
                [
                    [i1, 0, 0, -i2, 0],
                    [0, i1, 0, 0, -i2],
                    [0, 0, i1, 0, 0],
                    [-i2, 0, 0, i3, 0],
                    [0, -i2, 0, 0, i3],
                ]
            )

        # 3. Shape Function Evaluation and Skew Mapping
        # ---------------------------------------------------------------------
        len_u, len_v, len_w = (
            len(self.base_u),
            len(self.base_v),
            len(self.base_w),
        )
        len_bx, len_by = len(self.base_bx), len(self.base_by)
        self.len_tot = len_u + len_v + len_w + len_bx + len_by

        wgs = self.mesh[:, -1]

        # Mapping derivatives from local (xi, eta) to physical (x, y)
        dedx = 2.0 / self.a
        dedy = -2.0 / self.a * np.tan(self.beta)
        dndy = 2.0 / (self.b * np.cos(self.beta))

        xys = np.array(self.mesh[:, [0, 1]])

        # Helper to fetch and derive functions
        def get_basis(base_set, attr1, attr2):
            return Function.evaluate_basis_product(
                base_set, xys, self.basis_type, attr1, attr2
            )

        # Shape functions for u
        nu = get_basis(self.base_u, "vfunc", "vfunc")
        nu_e = get_basis(self.base_u, "vdfunc", "vfunc")
        nu_n = get_basis(self.base_u, "vfunc", "vdfunc")

        # Shape functions for v
        nv = get_basis(self.base_v, "vfunc", "vfunc")
        nv_e = get_basis(self.base_v, "vdfunc", "vfunc")
        nv_n = get_basis(self.base_v, "vfunc", "vdfunc")

        # Shape functions for w
        nw = get_basis(self.base_w, "vfunc", "vfunc")
        nw_e = get_basis(self.base_w, "vdfunc", "vfunc")
        nw_n = get_basis(self.base_w, "vfunc", "vdfunc")

        # Shape functions for beta_x
        nbx = get_basis(self.base_bx, "vfunc", "vfunc")
        nbx_e = get_basis(self.base_bx, "vdfunc", "vfunc")
        nbx_n = get_basis(self.base_bx, "vfunc", "vdfunc")

        # Shape functions for beta_y
        nby = get_basis(self.base_by, "vfunc", "vfunc")
        nby_e = get_basis(self.base_by, "vdfunc", "vfunc")
        nby_n = get_basis(self.base_by, "vfunc", "vdfunc")

        # Zero matrices for padding
        zeros = np.zeros_like(nu)

        if self.theory == StructuralTheory.KIRCHHOFF:
            nbx = nby = nbx_e = nby_e = nbx_n = nby_n = zeros

        # Assemble Generalized N matrices
        nm = np.block(
            [
                [nu, zeros, zeros, zeros, zeros],
                [zeros, nv, zeros, zeros, zeros],
                [zeros, zeros, nw, zeros, zeros],
                [zeros, zeros, zeros, nbx, zeros],
                [zeros, zeros, zeros, zeros, nby],
            ]
        )

        nm_e = np.block(
            [
                [nu_e, zeros, zeros, zeros, zeros],
                [zeros, nv_e, zeros, zeros, zeros],
                [zeros, zeros, nw_e, zeros, zeros],
                [zeros, zeros, zeros, nbx_e, zeros],
                [zeros, zeros, zeros, zeros, nby_e],
            ]
        )

        nm_n = np.block(
            [
                [nu_n, zeros, zeros, zeros, zeros],
                [zeros, nv_n, zeros, zeros, zeros],
                [zeros, zeros, nw_n, zeros, zeros],
                [zeros, zeros, zeros, nbx_n, zeros],
                [zeros, zeros, zeros, zeros, nby_n],
            ]
        )

        # 4. Skew Transformation and Strain Matrices (B)
        # ---------------------------------------------------------------------
        self.t_skew = np.array(
            [
                [1, np.sin(self.beta), 0, 0, 0],
                [0, np.cos(self.beta), 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, np.cos(self.beta), 0],
                [0, 0, 0, -np.sin(self.beta), 1],
            ]
        )

        up = self.t_skew @ nm
        up_x = self.t_skew @ nm_e * dedx
        up_y = self.t_skew @ (nm_e * dedy + nm_n * dndy)

        # Membrane Strains
        ex = up_x[:, [0], : len_u + len_v]
        ey = up_y[:, [1], : len_u + len_v]
        gxy = up_y[:, [0], : len_u + len_v] + up_x[:, [1], : len_u + len_v]

        epsm = np.zeros(((len(self.mesh), 3, len_u + len_v)))
        epsm[:, [0], :] = ex
        epsm[:, [1], :] = ey
        epsm[:, [2], :] = gxy

        # Curvature and Bending
        idx_w = slice(len_u + len_v, len_u + len_v + len_w)
        idx_rot = slice(len_u + len_v + len_w, None)

        if self.theory == StructuralTheory.REISSNER_MINDLIN:
            kappa = np.zeros((len(self.mesh), 3, len_bx + len_by))
            kappa[:, 0, :] = up_x[:, 3, idx_rot]
            kappa[:, 1, :] = up_y[:, 4, idx_rot]
            kappa[:, 2, :] = up_y[:, 3, idx_rot] + up_x[:, 4, idx_rot]

            gamma = np.zeros(
                (len(self.mesh), 2, self.len_tot - (len_u + len_v))
            )
            gamma[:, 0, :] = (
                up_y[:, 2, len_u + len_v :] + up[:, 4, len_u + len_v :]
            )
            gamma[:, 1, :] = (
                up_x[:, 2, len_u + len_v :] + up[:, 3, len_u + len_v :]
            )

            b1 = np.zeros((len(self.mesh), 8, self.len_tot))
            b1[:, :3, : len_u + len_v] = epsm
            b1[:, 3:6, idx_rot] = kappa
            b1[:, 6:, len_u + len_v :] = gamma

        else:  # KF Theory Curvatures (using 2nd derivatives)
            nw_ee = get_basis(self.base_w, "vd2func", "vfunc")
            nw_nn = get_basis(self.base_w, "vfunc", "vd2func")
            nw_en = get_basis(self.base_w, "vdfunc", "vdfunc")

            nw_xx = nw_ee * (dedx**2)
            nw_yy = (
                nw_ee * (dedy**2) + nw_nn * (dndy**2) + 2 * nw_en * dedy * dndy
            )
            nw_xy = nw_ee * dedx * dedy + nw_en * dedx * dndy

            kappa = np.zeros((len(self.mesh), 3, len_w))
            kappa[:, [0], :] = nw_xx
            kappa[:, [1], :] = nw_yy
            kappa[:, [2], :] = 2 * nw_xy

            b1 = np.zeros((len(self.mesh), 6, len_u + len_v + len_w))
            b1[:, :3, : len_u + len_v] = epsm
            b1[:, 3:, idx_w] = kappa

        # Shell Curvature Contribution (R): only in y axis (shallow shell)
        if self.radius > 0:
            epsk = (1.0 / self.radius) * up[:, [2], idx_w]
            b1[:, [1], idx_w] = epsk

        # 5. Global Matrix Assembly (K, M, Aerodynamics)
        # ---------------------------------------------------------------------
        # K = Sum(w * B.T * C * B) * Jacobian
        self._K = (
            np.tensordot(
                wgs, np.transpose(b1, [0, 2, 1]) @ self.c0 @ b1, [0, 0]
            )
            * self.jac
        )

        # Mass Matrix M
        bdof = up
        if self.theory == StructuralTheory.KIRCHHOFF:
            bdof[:, 3, :] = up_x[:, 2, :]
            bdof[:, 4, :] = up_y[:, 2, :]
            bdof = bdof[:, :, : len_u + len_v + len_w]

        self._M = (
            np.tensordot(
                wgs, np.transpose(bdof, [0, 2, 1]) @ self.i0 @ bdof, [0, 0]
            )
            * self.jac
        )

        # Initially: Structural damping assumed zero
        self._C = np.zeros_like(self._M)

        # Aerodynamic and Auxiliary Matrices
        k1 = np.zeros((len(self.mesh), 5, self.len_tot))
        k1[:, 2, idx_w] = up[:, 2, idx_w]

        k2 = np.zeros_like(k1)
        k2[:, 2, idx_w] = up_x[:, 2, idx_w]

        k3 = np.zeros_like(k1)
        k3[:, 2, idx_w] = up_y[:, 2, idx_w]

        self._K_aer = (
            np.tensordot(wgs, np.transpose(k1, [0, 2, 1]) @ k2, [0, 0])
            * self.jac
        )
        self._K_aery = (
            np.tensordot(wgs, np.transpose(k1, [0, 2, 1]) @ k3, [0, 0])
            * self.jac
        )
        self._C_caer = (
            np.tensordot(wgs, np.transpose(k1, [0, 2, 1]) @ k1, [0, 0])
            * self.jac
        )

        self._K_aer_r = (
            np.tensordot(wgs, np.transpose(k1, [0, 2, 1]) @ k1, [0, 0])
            * self.jac
        )

        # Symmetrize matrices to mitigate numerical issues
        # self._M = (self._M.T + self._M) / 2.0
        # self._K = (self._K.T + self._K) / 2.0
        # self._C = (self._C.T + self._C) / 2.0

        # Thermal Stress matrices placeholders and spring stiffness matrices
        self._K_sig = np.zeros_like(self._K)
        self._K_spring = np.zeros_like(self._K)
        self._K_theta = np.zeros_like(self._K)

        # Save B matrices for recovery or non-linear terms for thermal stress
        self.b1 = b1
        self.bdof = bdof

        bt = np.zeros((len(self.mesh), 2, self.len_tot))
        bt[:, [0], len_u + len_v : len_u + len_v + len_w] = up_x[
            :, [2], len_u + len_v : len_u + len_v + len_w
        ]
        bt[:, [1], len_u + len_v : len_u + len_v + len_w] = up_y[
            :, [2], len_u + len_v : len_u + len_v + len_w
        ]
        self.bt = bt

        # Compatibility with skew transformation matrix
        if self.theory == StructuralTheory.KIRCHHOFF:
            self.t_skew = np.array(
                [
                    [1, np.sin(self.beta), 0],
                    [0, np.cos(self.beta), 0],
                    [0, 0, 1],
                ]
            )

        # Re-evaluate Torsional Edge Springs
        for side, k_val in self.torsional_springs.items():
            if k_val > 0:
                self._eval_torsional_edge(side, k_val)

        # Re-evaluate Point Linear Springs
        for spring in self.point_springs:
            coords = spring["coords"]  # [xi, eta]
            k_s = spring["k"]
            self._eval_linear_spring(coords, k_s)

    def compute_free_modes(self):
        """Calculate natural frequencies and mode shapes of the panel.

        Solves the generalized eigenvalue problem [K]{u} = lambda [M]{u}.
        The results are sorted by frequency and stored in Hz.
        """
        # Summing all stiffness and mass contributions
        k_total = self.K_structural
        m_total = self.M_global

        # Generalized eigenvalue problem
        eigvals, eigvecs = la.eig(k_total, m_total)

        # Natural frequencies in rad/s
        omega_rad = np.real(np.sqrt(eigvals))

        # Sort frequencies and corresponding eigenvectors
        sort_indices = omega_rad.argsort()
        omega_rad = omega_rad[sort_indices]
        eigvecs = eigvecs[:, sort_indices]

        # Store frequencies in Hz and mode shapes
        self.free_omega_hz = omega_rad / (2.0 * np.pi)
        self.eigvecs = eigvecs

    def plot_free_modes(self, n_modes: int, axis_off: bool = True):
        """Plot the structural free vibration mode shapes in 3D.

        This method visualizes the natural frequencies and mode shapes using
        a 3D trisurf plot. It requires the modal analysis to have been
        previously executed.

        Args:
            n_modes (int): Number of mode shapes to visualize.
            axis_off (bool): If True, hides the 3D axes and grid.
                             Default is True.

        Raises:
            AttributeError: If free vibration modes have not been computed.
        """
        # Check if modal data is available before proceeding
        if not hasattr(self, "free_omega_hz") or self.eigvecs is None:
            raise AttributeError(
                "Modal data not found. Please run self.compute_free_modes() "
                "before attempting to plot mode shapes."
            )

        # Define figure size based on the number of plots
        if n_modes == 1:
            fig = plt.figure(figsize=(6.5, 3.5), dpi=300)
        else:
            rows = int(np.ceil(n_modes / 2))
            fig = plt.figure(figsize=(10, 3.5 * rows), dpi=300)

        # Coordinate transformation (from natural to physical system)
        x_bar = (self.mesh[:, 0] + 1) * self.a / 2
        y_bar = (self.mesh[:, 1] + 1) * self.b / 2
        x_phys = x_bar + y_bar * np.sin(self.beta)
        y_phys = y_bar * np.cos(self.beta)

        # Initial transverse displacement for curved panels (shell theory)
        w0 = np.zeros(shape=y_phys.shape)
        if self.radius > 0:
            alpha = self.b / (2 * self.radius)
            w0 = self.radius * (
                np.cos((y_phys / self.radius) - alpha) - np.cos(alpha)
            )

        for i in range(min(n_modes, self.eigvecs.shape[1])):
            # Extract displacement mode (z-direction DOF)
            # Bdof maps generalized Ritz coordinates to physical grid
            mode_shape = (self.bdof @ (self.eigvecs[:, [i]]))[:, 2, 0]

            # Normalize mode shape for visualization (peak always at +1)
            max_val = np.max(np.abs(mode_shape))
            if np.max(mode_shape) == max_val:
                mode_shape = mode_shape / max_val
            else:
                mode_shape = -mode_shape / max_val

            # Superimpose on initial curvature for shell-like structures
            if self.radius > 0:
                mode_shape = mode_shape * np.max(w0) + w0

            # Subplot positioning logic
            if n_modes == 1:
                ax = fig.add_subplot(1, 1, 1, projection="3d")
            else:
                ax = fig.add_subplot(
                    int(np.ceil(n_modes / 2)), 2, i + 1, projection="3d"
                )

            # Title formatting with LaTeX using the updated variable name
            ax.set_title(
                f"${i + 1}^o$ mode, $f={self.free_omega_hz[i]:.4f}$ Hz",
                fontsize=11,
            )

            # Surface plotting
            ax.plot_trisurf(x_phys, y_phys, mode_shape, cmap=cm.jet)

            # Labeling and physical limits
            ax.set_xlabel("$x$ [m]")
            ax.set_ylabel("$y$ [m]")
            ax.set_zlabel("$w$ [-]")

            x_limit = self.a + self.b * np.sin(self.beta)
            y_limit = self.b * np.cos(self.beta)
            ax.set_xlim(0, x_limit)
            ax.set_ylim(0, y_limit)

            # Visual aspect ratio (30% height for better mode visualization)
            ax.set_box_aspect((1, self.b / self.a, 0.3))
            if axis_off:
                ax.set_axis_off()

        plt.tight_layout()
        plt.show(block=False)

    def plot_free_mode(self, mode_idx: int, axis_off: bool = True):
        """Plot a single structural mode shape in 3D.

        This method visualizes a specific natural frequency and its
        corresponding mode shape. It requires the modal analysis results to be
        available in the object.

        Args:
            mode_idx (int): The index of the mode to plot (1-based index).
            axis_off (bool): If True, hides the 3D axes and grid.
                             Default is True.

        Raises:
            AttributeError: If modal analysis has not been performed.
            IndexError: If the requested mode index is out of range.
        """
        # Ensure modal data is present
        if not hasattr(self, "free_omega_hz") or self.eigvecs is None:
            raise AttributeError(
                "Modal data not found. Run self.free_modes() before plotting."
            )

        # Validate mode index (converting 1-based user input to 0-based index)
        if mode_idx < 1 or mode_idx > self.eigvecs.shape[1]:
            raise IndexError(
                f"Mode index {mode_idx} is out of range. "
                f"Available modes: 1 to {self.eigvecs.shape[1]}."
            )

        fig = plt.figure(figsize=(6.5, 3.5), dpi=300)

        # Physical coordinate transformation
        x_bar = (self.mesh[:, 0] + 1) * self.a / 2
        y_bar = (self.mesh[:, 1] + 1) * self.b / 2
        x_phys = x_bar + y_bar * np.sin(self.beta)
        y_phys = y_bar * np.cos(self.beta)

        # Base geometry for curved panels
        w0 = np.zeros(shape=y_phys.shape)
        if self.radius > 0:
            alpha = self.b / (2 * self.radius)
            w0 = self.radius * (
                np.cos((y_phys / self.radius) - alpha) - np.cos(alpha)
            )

        # Extract and normalize mode shape
        # Adjusting index to 0-based (mode_idx - 1)
        mode_shape = (self.bdof @ (self.eigvecs[:, [mode_idx - 1]]))[:, 2, 0]

        abs_max = np.max(np.abs(mode_shape))
        if np.max(mode_shape) == abs_max:
            mode_shape = mode_shape / abs_max
        else:
            mode_shape = -mode_shape / abs_max

        # Add initial curvature for shells
        if self.radius > 0:
            mode_shape = mode_shape * np.max(w0) + w0

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_title(
            f"${mode_idx}^o$ mode,"
            + f" $f={self.free_omega_hz[mode_idx - 1]:.4f}$ Hz",
            fontsize=11,
        )

        # Visualization
        ax.plot_trisurf(x_phys, y_phys, mode_shape, cmap=cm.jet)

        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_zlabel("$w$ [-]")

        # Axis limits and aspect ratio
        ax.set_xlim(0, self.a + self.b * np.sin(self.beta))
        ax.set_ylim(0, self.b * np.cos(self.beta))
        ax.set_box_aspect((1, self.b / self.a, 0.3))

        if axis_off:
            ax.set_axis_off()

        plt.show(block=False)

    def compute_rayleigh_damping(
        self, omega_low: float = 0.0, omega_high: float = 0.0
    ):
        """Determine the Rayleigh damping matrix [C] = alpha*[M] + beta*[K].

        If frequency bounds are not provided, the method uses the first and
        last computed natural frequencies. Damping ratio (zeta) is retrieved
        from the laminate material properties.

        Args:
            omega_low (float, optional): Lower bound frequency [rad/s].
            omega_high (float, optional): Upper bound frequency [rad/s].
        """
        # Automatically compute modes if frequencies are not provided
        if omega_low == 0.0 or omega_high == 0.0:
            if not hasattr(self, "free_omega_hz"):
                self.compute_free_modes()

            # Convert Hz back to rad/s for calculations
            omega_low = self.free_omega_hz[0] * 2.0 * np.pi
            omega_high = self.free_omega_hz[-1] * 2.0 * np.pi

        # System of equations for Rayleigh coefficients:
        # [1, omega_i^2] * {a} = {2 * zeta * omega_i}
        # Resulting in alpha = a1 and beta = a2
        mat_a = np.array([[1.0, omega_low**2], [1.0, omega_high**2]])

        # Accessing zeta from the first ply's material
        zeta = self.laminate.plies[0].material.damping
        vec_b = np.array([[2.0 * zeta * omega_low], [2.0 * zeta * omega_high]])

        # Solving for a1 (mass prop) and a2 (stiffness prop)
        alpha_beta = np.linalg.solve(mat_a, vec_b)

        # Global structural damping matrix assembly
        self._C = alpha_beta[0] * self._M + alpha_beta[1] * self._K

    def compute_thermal_stiffness(self, delta_t: float, **kwargs):
        """Compute the thermal stress stiffness matrix (geometric stiffness).

        This method solves the equivalent static problem for a temperature
        change (Delta T), calculates the resulting internal loads (N0),
        and assembles the incremental stiffness matrix K_sig.

        Args:
            delta_t (float): Temperature variation.
            **kwargs:
                distribution (str): 'cte' for uniform or other for spatial.
        """
        # Re-evaluate mesh for non-linear integration (nlin=1)
        self._initialize_quadrature(nlin=1)

        # Reset thermal stiffness
        self._K_sig = np.zeros((self.len_tot, self.len_tot))

        distribution = kwargs.get("distribution", "cte")
        weights = self.mesh[:, -1]

        if distribution == "cte":
            nr_aux = self.laminate.N_R * delta_t
            mr_aux = self.laminate.M_R * delta_t
        else:
            # Spatial distribution based on predifined function
            xys = self.mesh[:, [0, 1]]
            nt_dist = Function.evaluate_basis_product(
                [[1, 1]], xys, distribution, "vfunc", "vfunc"
            )

            nt_scaled = nt_dist * delta_t
            nr_aux = nt_scaled * self.laminate.N_R
            mr_aux = nt_scaled * self.laminate.M_R

        # Assemble equivalent thermal load vector (NM_R)
        nm_r = np.block([[nr_aux], [mr_aux]])

        if self.theory == StructuralTheory.REISSNER_MINDLIN:
            # Add zeros for shear components in FSDT
            zeros_shear = np.zeros((len(self.mesh), 2, 1))
            nm_r = np.block([[nm_r], [zeros_shear]])

        # Compute equivalent thermal force vector (F_tilde)
        f_tilde = (
            np.tensordot(
                weights, np.transpose(self.b1, [0, 2, 1]) @ nm_r, [0, 0]
            )
            * self.jac
        )

        # Solve for equivalent displacements using Cholesky decomposition
        c_factor = la.cho_factor(self.K_structural)
        q_t = la.cho_solve(c_factor, f_tilde)

        # Internal loads N0 and effective thermal loads N
        # N0 = C0 * B1 * qt
        n0 = self.c0 @ self.b1 @ q_t
        self.q_t = q_t
        self.n0 = n0

        # Resulting stress resultants for geometric stiffness
        n_eff = n0 - nm_r

        # Construct the Stress Resultant Matrix (NT) for integration
        # NT = [Nx, Nxy; Nxy, Ny]
        n_tensor = np.zeros((len(self.mesh), 2, 2))
        n_tensor[:, [0], [0]] = n_eff[:, [0], [0]]  # Nx
        n_tensor[:, [0], [1]] = n_eff[:, [2], [0]]  # Nxy
        n_tensor[:, [1], [0]] = n_eff[:, [2], [0]]  # Nxy
        n_tensor[:, [1], [1]] = n_eff[:, [1], [0]]  # Ny

        # Assemble Geometric Stiffness Matrix: Integral(BT.T * NT * BT)
        K_sig = (
            np.tensordot(
                weights,
                np.transpose(self.bt, [0, 2, 1]) @ n_tensor @ self.bt,
                [0, 0],
            )
            * self.jac
        )

        # Symmetrize to mitigate numerical issues
        # K_sig = (K_sig.T + K_sig) / 2.0

        self._K_sig = K_sig

    def add_torsional_edge(self, side: str, k_theta: float):
        """Apply a torsional spring stiffness to a specific edge of the panel.

        Updates the edge definition in the storage dictionary and evaluates
        the line integral for the stiffness contribution.

        Args:
            side (str): Edge identifier ('left', 'right', 'bottom', 'top').
            k_theta (float): Rotational stiffness constant.
        """
        side = side.lower()
        if side not in self.torsional_springs:
            raise ValueError(f"Invalid side identifier: {side}")

        # 1. Store/Overwrite the physical definition
        self.torsional_springs[side] = k_theta

        # 2. Trigger the line integration engine
        self._eval_torsional_edge(side, k_theta)

    def _eval_torsional_edge(self, side: str, k_theta: float):
        """Apply a torsional spring stiffness to a specific edge of the panel.

        This effectively models semi-rigid boundary conditions by adding to the
        stiffness matrix K_theta.

        Args:
            side (str): Edge identifier ('left', 'right', 'bottom', 'top').
            k_theta (float): Rotational stiffness constant.
        """
        side = side.lower()
        xi_ones = np.ones_like(self.xi)

        # 1. Define integration points and weights along the selected edge
        if side in ["left", "xi_min"]:
            points = np.column_stack([-1.0 * xi_ones, self.xi])
        elif side in ["right", "xi_max"]:
            points = np.column_stack([xi_ones, self.xi])
        elif side in ["bottom", "eta_min"]:
            points = np.column_stack([self.xi, -1.0 * xi_ones])
        elif side in ["top", "eta_max"]:
            points = np.column_stack([self.xi, xi_ones])
        else:
            raise ValueError(f"Invalid side: {side}")

        weights = self.wi
        dedx = 2.0 / self.a
        dedy = -2.0 / self.a * np.tan(self.beta)
        dndy = 2.0 / (self.b * np.cos(self.beta))

        # 2. Evaluate basis functions at the edge
        if self.theory == StructuralTheory.KIRCHHOFF:
            # Kirchhoff uses w derivatives for rotation
            nw_e = Function.evaluate_basis_product(
                self.base_w, points, self.basis_type, "vdfunc", "vfunc"
            )
            nw_n = Function.evaluate_basis_product(
                self.base_w, points, self.basis_type, "vfunc", "vdfunc"
            )

            # Map rotations to physical coordinates
            nw_x = nw_e * dedx
            nw_y = nw_e * dedy + nw_n * dndy

            theta = (
                nw_x if side in ["left", "right", "xi_min", "xi_max"] else nw_y
            )

            # Assembly for Kirchhoff (only u, v, w exist)
            zeros = np.zeros_like(theta)
            b_theta = np.block(
                [
                    [zeros, zeros, theta],
                ]
            )

        elif self.theory == StructuralTheory.REISSNER_MINDLIN:
            # RM uses beta_x and beta_y directly
            nbx = Function.evaluate_basis_product(
                self.base_bx, points, self.basis_type, "vfunc", "vfunc"
            )
            nby = Function.evaluate_basis_product(
                self.base_by, points, self.basis_type, "vfunc", "vfunc"
            )

            # Build rotation block for displacement vector mapping
            nm = np.block(
                [
                    [zeros, zeros, zeros, zeros, zeros],
                    [zeros, zeros, zeros, zeros, zeros],
                    [zeros, zeros, zeros, zeros, zeros],
                    [zeros, zeros, zeros, nbx, zeros],
                    [zeros, zeros, zeros, zeros, nby],
                ]
            )
            # The above is simplified to focus on rotation DOFs
            up = self.t_skew @ nm

            # Select beta_x (index 3) or beta_y (index 4) based on edge
            if side in ["left", "right", "xi_min", "xi_max"]:
                b_theta = up[:, [3], :]
            else:
                b_theta = up[:, [4], :]

        # 3. Integrate torsional stiffness:
        if side in ["bottom", "top", "eta_min", "eta_max"]:
            edge_length = self.a
            line_jacobian = edge_length / 2.0
        else:
            edge_length = self.b
            line_jacobian = edge_length / 2.0
        k_incremental = k_theta * (
            np.tensordot(
                weights, np.transpose(b_theta, [0, 2, 1]) @ b_theta, [0, 0]
            )
            * line_jacobian
        )

        # Symmetrize to mitigate numerical issues
        # k_incremental = (k_incremental.T + k_incremental) / 2.0

        # Add to existing torsional stiffness contributions
        self._K_theta += k_incremental

    def remove_torsional_springs(self, edge: str = "all"):
        """Remove torsional springs from specific edges or all of them.

        Args:
            edge (str): The edge to remove ('left', 'right', 'bottom', 'top').
                Defaults to 'all'.
        """
        if edge == "all":
            for key in self.torsional_springs:
                self.torsional_springs[key] = 0.0
                self._K_theta *= 0.0
        elif edge in self.torsional_springs:
            self.torsional_springs[edge] = 0.0
            self._K_theta *= 0.0
            # Re-evaluate remaining Torsional Edge Springs
            for side, k_val in self.torsional_springs.items():
                if k_val > 0:
                    self._eval_torsional_edge(side, k_val)

    def add_linear_spring(self, coords: np.ndarray, k: float):
        """Register a new linear transverse spring at a specific point.

        This method stores the spring definition and immediately computes its
        stiffness contribution to the global system.

        Args:
            coords (np.ndarray): Natural coordinates (xi, eta).
            k (float): Stiffness constant of the spring.
        """
        # 1. Store the definition for persistence (for model rebuilding)
        self.point_springs.append({"coords": np.array(coords), "k": k})

        # 2. Trigger the incremental evaluation
        self._eval_linear_spring(coords, k)

    def _eval_linear_spring(self, point: np.ndarray, k_spring: float):
        """Add a discrete linear spring at a specific (xi, eta) location.

        This adds a point stiffness to the K_spring matrix.

        Args:
            point (np.ndarray): Coordinate array [xi, eta].
            k_spring (float): Linear stiffness constant.
        """
        point_2d = np.atleast_2d(point)
        # Evaluate w shape functions at the specific point
        nw = Function.evaluate_basis_product(
            self.base_w, point_2d, self.basis_type, "vfunc", "vfunc"
        )

        zeros = np.zeros_like(nw)

        if self.theory == StructuralTheory.KIRCHHOFF:
            # Assembly for KF: [zeros_u, zeros_v, Nw]
            b_w = np.block([zeros, zeros, nw])

        elif self.theory == StructuralTheory.REISSNER_MINDLIN:
            b_w = np.block([zeros, zeros, nw, zeros, zeros])

        # Discrete stiffness contribution: k * {Bw}.T @ {Bw}
        k_discrete = k_spring * (np.transpose(b_w, [0, 2, 1]) @ b_w)

        # Symmetrize to mitigate numerical issues
        # k_discrete = (k_discrete.T + k_discrete) / 2.0

        # Accumulate into the global spring stiffness matrix
        self._K_spring += k_discrete[0, :, :]

    def remove_point_springs(self, coords: tuple = None):
        """Remove a specific point spring by coordinates or all of them.

        Args:
            coords (tuple, optional): A tuple (x, y) identifying the spring
                to be removed. If None, the entire list is cleared.
        """
        if coords is None:
            self.point_springs = []
            self._K_spring *= 0.0
        else:
            # Filter list by comparing the 'coords' key with the input tuple
            self.point_springs = [
                s
                for s in self.point_springs
                if not np.allclose(s["coords"], np.array(coords))
            ]
            self._K_spring *= 0.0
            # Re-evaluate remaining Linear Springs
            for spring in self.point_springs:
                coords = spring["coords"]  # [xi, eta]
                k_s = spring["k"]
                self._eval_linear_spring(coords, k_s)

        # Reset the associated stiffness matrix if it exists
        if hasattr(self, "_K_points"):
            self._K_points *= 0.0

    def insert_stiffener(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        height: float,
        laminate: Laminate,
        gap: float = 0.0,
        side: int = -1,
    ):
        """Insert a stiffener between two points defined in global coordinates.

        This method creates a new Stiffener object, computes its properties,
        and integrates its stiffness and mass contributions into the global
        system.

        Args:
            x0 (float): x-coordinate of the first point.
            y0 (float): y-coordinate of the first point.
            x1 (float): x-coordinate of the second point.
            y1 (float): y-coordinate of the second point.
            height (float): The vertical dimension of the stiffener.
            laminate (Laminate): The laminate defining the stiffener width.
            gap (float): Physical distance between panel surface and stiffener.
                Defaults to 0.0.
            side (int): 1 for top surface (positive z), -1 for bottom surface.
                Defaults to 1.
        """
        # Create a new Stiffener instance with the provided coordinates
        stiffener = Stiffener(
            x0, y0, x1, y1, height, laminate, self, gap=gap, side=side
        )

        # Add the stiffener to the panel's list of stiffeners
        self.stiffeners.append(stiffener)

    def remove_stiffeners(self, index: int = None):
        """Remove a specific stiffener by its index or all of them.

        Args:
            index (int, optional): The index of the stiffener in the list.
                If None, all stiffeners are removed.
        """
        if index is None:
            self.stiffeners = []
        else:
            try:
                self.stiffeners.pop(index)
            except IndexError:
                pass

    def run_thermal_buckling(
        self, plot_first_mode: bool = True, **kwargs
    ) -> float:
        """Compute the critical thermal buckling temperature.

        This method solves the generalized eigenvalue problem to find the
        lowest temperature variation (Delta T_cr) that triggers buckling,
        considering the potential spatial distribution of the temperature.

        Args:
            plot_first_mode (bool): If True, generates a 3D trisurf plot of
                the first buckling mode shape.
            **kwargs:
                distribution (str): The temperature field distribution type
                    (e.g., "cte" for constant, "sines", etc.). Defaults "cte".

        Returns:
            float: The critical temperature variation (Delta T_cr).
        """
        # Define the temperature distribution (default is constant)
        dist = kwargs.get("distribution", "cte")

        # Compute geometric thermal stiffness for a unit temperature variation.
        self.compute_thermal_stiffness(1.0, distribution=dist)

        # Solve the generalized eigenvalue problem:
        # [K_str - K_sig] {phi} = -lambda [K_sig] {phi}
        # Note: Using the structural and geometric stiffness matrices directly
        eigvals, eigvecs = la.eig(
            self.K_structural - self._K_sig, -self._K_sig
        )

        # Sort eigenvalues to find the first positive critical Delta T
        idx = np.real(eigvals).argsort()
        dt_cr = float(np.real(eigvals[idx[0]]))
        first_mode_vector = eigvecs[:, idx[0]]

        # Map reference coordinates to physical geometry (handling skewness)
        x_phys = (self.mesh[:, 0] + 1) * self.a / 2
        y_phys = (self.mesh[:, 1] + 1) * self.b / 2
        x_coords = x_phys + y_phys * np.sin(self.beta)
        y_coords = y_phys * np.cos(self.beta)

        # Map Degree of Freedom (DOF) vector to physical displacement (w)
        # Assuming index 2 represents the transverse displacement
        mode_shape = (self.bdof @ first_mode_vector)[:, 2]

        # Normalize the mode shape for consistent visualization
        mode_shape = mode_shape / np.max(np.abs(mode_shape))

        if plot_first_mode:
            # Initialize base displacement (zero for flat panels)
            w0 = np.zeros_like(y_coords)

            if self.radius > 0:
                # Calculate initial curvature geometry for curved panels
                # R is assumed to be the radius property of the panel
                alpha = self.b / (2 * self.radius)
                w0 = self.radius * (
                    np.cos((y_coords / self.radius) - alpha) - np.cos(alpha)
                )

                # Scale mode shape to match the physical scale of the curvature
                mode_shape = mode_shape * np.max(w0)
                mode_shape += w0

            # Generate the 3D plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")

            # Using trisurf for unstructured or mapped grids
            ax.plot_trisurf(x_coords, y_coords, mode_shape, cmap=cm.jet)

            # Formatting and labeling with LaTeX
            ax.set_title(
                rf"Buckling Mode at $\Delta T_{{cr}} = {dt_cr:.2f}$ °C"
            )
            ax.set_xlabel(r"$x$ $[m]$")
            ax.set_ylabel(r"$y$ $[m]$")
            ax.set_zlabel(r"$w$ $[m]$")

            # Set axis limits based on panel dimensions and skew angle
            ax.set_xlim(0, self.a + self.b * np.sin(self.beta))
            ax.set_ylim(0, self.b * np.cos(self.beta))

            # Set aspect ratio and hide axes for a clean presentation look
            ax.set_box_aspect((1, self.b / self.a, 0.3))
            ax.set_axis_off()

            plt.show()

        return dt_cr

    @property
    def K_structural(self) -> np.ndarray:
        """Sum of all internal stiffness components.

        Includes: baseline panel, thermal effects (K_sig), discrete
        springs, torsional edge springs, and all attached stiffeners.
        Note: Does not include aerodynamic stiffness (Kaer).
        """
        # Sum of panel's own stiffness terms
        k_str = self._K + self._K_sig + self._K_spring + self._K_theta

        # Add contributions from each stiffener
        if hasattr(self, "stiffeners") and self.stiffeners:
            for stf in self.stiffeners:
                k_str += stf._K

        return k_str

    @property
    def C_structural(self) -> np.ndarray:
        """Structural damping matrix (Rayleigh + Stiffeners).

        Note: Does not include aerodynamic damping (Caer).
        """
        c_str = self._C.copy()  # Rayleigh damping from panel

        if hasattr(self, "stiffeners") and self.stiffeners:
            for stf in self.stiffeners:
                if hasattr(stf, "C"):
                    c_str += stf._C

        return c_str

    @property
    def M_global(self) -> np.ndarray:
        """Global mass matrix including panel and all stiffeners.

        Since mass is a fundamental property and usually doesn't have
        'external' components like stiffness, 'global' remains appropriate.
        """
        m_glob = self._M.copy()

        if hasattr(self, "stiffeners") and self.stiffeners:
            for stf in self.stiffeners:
                m_glob += stf._M

        return m_glob

    def eval_bdof_at_point(self, point: np.array) -> np.array:
        """Calculate matrix that provides the displacements at a point.

        This method interpolates the global degrees of freedom (DOF)

        Args:
            point (np.array): Non-dimensional coordinates [xi, eta] where
                the displacement is evaluated.

        Returns:
            float: The shape functions DOF matrix.
        """
        # Extract non-dimensional coordinates and prepare evaluation point
        xp = point[0]
        yp = point[1]
        xys = np.array([[xp, yp]])

        # Helper to fetch and evaluate basis functions at the given point
        def get_basis(base_set, attr1, attr2):
            """Evaluate the product of basis functions for a given set."""
            return Function.evaluate_basis_product(
                base_set, xys, self.basis_type, attr1, attr2
            )

        # Evaluate shape functions for each field (u, v, w, bx, by)
        nu = get_basis(self.base_u, "vfunc", "vfunc")
        nv = get_basis(self.base_v, "vfunc", "vfunc")
        nw = get_basis(self.base_w, "vfunc", "vfunc")
        nbx = get_basis(self.base_bx, "vfunc", "vfunc")
        nby = get_basis(self.base_by, "vfunc", "vfunc")

        # Zero matrices for consistent block diagonal padding
        zeros = np.zeros_like(nu)

        # Assemble the Generalized Displacement Matrix (N)
        # Constructing a block diagonal matrix to map the panel
        if self.theory == StructuralTheory.KIRCHHOFF:
            nm = np.block(
                [
                    [nu, zeros, zeros],
                    [zeros, nv, zeros],
                    [zeros, zeros, nw],
                ]
            )
        else:
            nm = np.block(
                [
                    [nu, zeros, zeros, zeros, zeros],
                    [zeros, nv, zeros, zeros, zeros],
                    [zeros, zeros, nw, zeros, zeros],
                    [zeros, zeros, zeros, nbx, zeros],
                    [zeros, zeros, zeros, zeros, nby],
                ]
            )

        # Apply skew transformation matrix (t_skew) and map to DOF vector
        return self.t_skew @ nm

    def get_displacement_at_points(
        self, point: np.array, dof: int, qts: np.array
    ) -> float:
        """Calculate the generalized displacements at a specific coordinate.

        This method interpolates the global degrees of freedom (DOF) to find
        the physical displacements and rotations at a given point using
        the panel's basis functions and the skew transformation.

        Args:
            point (np.array): Non-dimensional coordinates [xi, eta] where
                the displacement is evaluated.
            dof (int): Index of the degree of freedom to return
                (e.g., 0 for u, 1 for v, 2 for w, 3 for bx, 4 for by).
            qts (np.array): Vector of generalized coordinates (DOF vector).

        Returns:
            float: The interpolated displacement at the requested
                point for the specified DOF.
        """
        # Extract non-dimensional coordinates and prepare evaluation point
        bdof = self.eval_bdof_at_point(point)

        # Apply skew transformation matrix (t_skew) and map to DOF vector
        # This handles the transformation from skewed to physical system
        up = (bdof @ qts)[0, :, :]
        # Return only the requested displacement component
        return up[dof, :]

    def init_nlin(self, *args, **kwargs):
        """Initialize tensors and matrices for non-linear structural analysis.

        This method pre-calculates the xpatial integration of shape function
        and strain-displacement relations required for Von Karman non-linear
        internal forces. It stores tensors that allow fast force evaluation
        during time integration.
        """
        # Initialize quadrature for non-linear terms
        self._initialize_quadrature(nlin=1)

        # Membrane and curvature strain-displacement matrices
        self.epsm = self.b1[:, :3, :]
        self.kappa = self.b1[:, 3:6, :]
        self.epsmT = np.transpose(self.epsm, [0, 2, 1])
        self.kappaT = np.transpose(self.kappa, [0, 2, 1])

        len_w = len(self.base_u)
        len_tot = self.len_tot

        # Geometric transformation for skewed coordinates
        dedx = 2.0 / self.a
        dedy = -2.0 / self.a * np.tan(self.beta)
        dndy = 2.0 / (self.b * np.cos(self.beta))

        xys = np.array(self.mesh[:, [0, 1]])

        def get_basis(base_set, attr1, attr2):
            """Evaluate basis product for derivatives."""
            return Function.evaluate_basis_product(
                base_set, xys, self.basis_type, attr1, attr2
            )

        # Derivatives of shape functions for transverse displacement 'w'
        nw_e = get_basis(self.base_w, "vdfunc", "vfunc")
        nw_n = get_basis(self.base_w, "vfunc", "vdfunc")

        lenXi = len(self.xi)
        len_nl = int((len_w + 1) * len_w / 2)

        # Pre-allocate non-linear interaction tensors
        self.NLNL = np.zeros((len_nl, len_nl))
        self.kNL = np.zeros((len_tot, len_nl))
        self.NLk = np.zeros((len_nl, len_tot))
        self.eNL = np.zeros((len_tot, len_nl))
        self.NLe = np.zeros((len_nl, len_tot))

        for ixi in range(lenXi):
            # Slice mesh for the current integration row
            slice_idx = slice(ixi * lenXi, (ixi + 1) * lenXi)

            nw_e_curr = nw_e[slice_idx, :, :]
            nw_n_curr = nw_n[slice_idx, :, :]
            wgs = self.mesh[slice_idx, -1]

            # Evaluation of spatial derivatives in skewed system
            Nw_x = nw_e_curr * dedx
            Nw_y = nw_e_curr * dedy + nw_n_curr * dndy

            # Nbar matrices
            Nbarw_x = np.zeros((lenXi, len_w, len_nl))
            Nbarw_y = np.zeros((lenXi, len_w, len_nl))

            # Optimization: Use vectorization to fill the diagonal and
            # upper blocks
            for km in range(lenXi):
                Nbarw_x[km, :, :len_w] = np.diag(Nw_x[km, 0, :])
                Nbarw_y[km, :, :len_w] = np.diag(Nw_y[km, 0, :])

                for kN in range(len_w):
                    # Indexing for the symmetric triangular part
                    start = int((1 + kN) * len_w - (1 + kN) * kN / 2)
                    end = int((2 + kN) * len_w - (1 + kN) * (2 + kN) / 2)

                    # Fill cross-terms for non-linear products
                    Nbarw_x[km, kN, start:end] = Nw_x[km, 0, kN + 1 :]
                    Nbarw_x[km, kN + 1 :, start:end] = Nw_x[
                        km, 0, kN
                    ] * np.eye(len_w - (1 + kN))

                    Nbarw_y[km, kN, start:end] = Nw_y[km, 0, kN + 1 :]
                    Nbarw_y[km, kN + 1 :, start:end] = Nw_y[
                        km, 0, kN
                    ] * np.eye(len_w - (1 + kN))

            # Assemble non-linear shape function blocks
            NNbarw_x = Nw_x @ Nbarw_x
            NNbarw_y = Nw_y @ Nbarw_y
            NNbarw_xy = Nw_x @ Nbarw_y

            # Strain interpolation matrix (Nbar) for quadratic terms
            Nbar = 0.5 * np.block([[NNbarw_x], [NNbarw_y], [2 * NNbarw_xy]])
            dNbar = np.block([[NNbarw_x], [NNbarw_y], [2 * NNbarw_xy]])
            dNbT = np.transpose(dNbar, [0, 2, 1])

            # Local slices of strain-displacement matrices
            kapa = self.kappa[slice_idx, :, :]
            kapaT = self.kappaT[slice_idx, :, :]
            epsm = self.epsm[slice_idx, :, :]
            epsmT = self.epsmT[slice_idx, :, :]

            # Integrate non-linear stiffness components using quadrature
            # [A], [B] matrices from laminate (Classical Plate Theory)
            self.NLNL += (
                np.tensordot(wgs, dNbT @ self.laminate.A @ Nbar, [0, 0])
                * self.jac
            )
            self.kNL += (
                np.tensordot(wgs, kapaT @ self.laminate.B @ Nbar, [0, 0])
                * self.jac
            )
            self.NLk += (
                np.tensordot(wgs, dNbT @ self.laminate.B @ kapa, [0, 0])
                * self.jac
            )
            self.eNL += (
                np.tensordot(wgs, epsmT @ self.laminate.A @ Nbar, [0, 0])
                * self.jac
            )
            self.NLe += (
                np.tensordot(wgs, dNbT @ self.laminate.A @ epsm, [0, 0])
                * self.jac
            )

    def compute_nl(self, ut):
        """Compute the non-linear internal force vector.

        Calculates the internal forces arising from large deflections
        (Von Kármán strains), accounting for membrane-bending coupling
        and stiffness from stiffeners.

        Args:
            ut (ndarray): Global displacement vector (or modal coordinates).

        Returns:
            ndarray: The non-linear internal force vector.
        """
        len_u, len_v, len_w = (
            len(self.base_u),
            len(self.base_v),
            len(self.base_w),
        )

        # Extract transverse displacement degrees of freedom
        u3 = ut[len_u + len_v : len_u + len_v + len_w]
        len_nl = int((len_w + 1) * len_w / 2)

        # Nu3bar maps u3 to the non-linear vector space
        Nu3bar = np.zeros((len_nl, len_w))

        # Fill diagonal components (u_i * u_i terms)
        Nu3bar[:len_w, :] = np.diag(u3[:, 0])

        # Fill cross-product terms (u_i * u_j terms)
        for kk, uu in enumerate(u3):
            start = int((1 + kk) * len_w - (1 + kk) * kk / 2)
            end = int((2 + kk) * len_w - (1 + kk) * (2 + kk) / 2)

            # Triangular interaction mapping
            Nu3bar[start:end, kk] = 0.5 * u3[kk + 1 :, 0]
            Nu3bar[start:end, kk + 1 :] = 0.5 * uu * np.eye(len_w - (1 + kk))

        # Assemble global mapping matrix
        Nu3q = np.zeros((len_nl, self.len_tot))
        Nu3q[:, len_u + len_v : len_u + len_v + len_w] = Nu3bar

        # Purely non-linear force (cubic in displacement)
        # FNL = {q}^T * [NLNL] * {q}
        FNL = Nu3q.T @ self.NLNL @ Nu3q @ ut

        # Membrane-Bending coupling forces (A-B coupling)
        # Includes curvature and membrane interactions
        F_acp = (
            (self.kNL @ Nu3q @ ut)
            + (self.eNL @ Nu3q @ ut)
            + (Nu3q.T @ self.NLk @ ut)
            + (Nu3q.T @ self.NLe @ ut)
        )

        # Stiffener contributions
        F_stf = np.zeros_like(FNL)
        for stf in self.stiffeners:
            F_stf += stf.compute_nl(ut)

        return FNL + F_acp + F_stf
