"""Stiffener module for laminate panels.

This module provides the Stiffener class for representing rectangular
stiffeners with laminate properties oriented perpendicular to panels.
"""

import numpy as np

from .basis_functions import Function
from .definitions import StructuralTheory
from .material import Laminate


class Stiffener:
    """Beam model for laminate stiffener with rectangular cross section.

    Calculates stiffness, mass, and thermal matrices using
    numerical integration (Gauss-Legendre) along the stiffener length.
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        height: float,
        laminate: Laminate,
        panel,
        gap: float = 0.0,
        side: int = -1,
    ):
        """Initialize the Stiffener object.

        Args:
            x0 (float): Starting x-coordinate in the panel plane.
            y0 (float): Starting y-coordinate in the panel plane.
            x1 (float): Ending x-coordinate in the panel plane.
            y1 (float): Ending y-coordinate in the panel plane.
            height (float): The vertical dimension (depth) of the stiffener.
            laminate (Laminate): The laminate defining the stiffener width.
            panel (Panel): Object that satisfies the panel interface.
            gap (float): Physical distance between panel surface and stiffener.
                Defaults to 0.0.
            side (int): 1 for top surface (positive z), -1 for bottom surface.
                Defaults to -1.
        """
        # Geometric properties
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.height = height
        self.side = side
        self.gap = gap
        self.eccentricity = self.side * (
            (panel.laminate.total_thickness / 2) + gap + (self.height / 2)
        )

        # Width is derived from the laminate total thickness
        self.laminate = laminate
        self.width = laminate.total_thickness

        # Derived geometric attributes
        self.length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        self.theta = np.arccos((x1 - x0) / self.length)
        self.jac = self.length / 2

        # Assign basis functions from panel to stiffener for matrix evaluation
        self.base_u = panel.base_u
        self.base_v = panel.base_v
        self.base_w = panel.base_w
        self.base_bx = panel.base_bx
        self.base_by = panel.base_by

        # Map panel theory to corresponding beam theory
        if panel.theory == StructuralTheory.REISSNER_MINDLIN:
            self.theory = StructuralTheory.TIMOSHENKO
        elif panel.theory == StructuralTheory.KIRCHHOFF:
            self.theory = StructuralTheory.EULER_BERNOULLI

        # Cross-section properties (rectangular)
        self.area = self.height * self.width
        self.izz_s = (self.height * self.width**3) / 12.0
        self.inn_s = (self.width * self.height**3) / 12.0
        self.j0_s = self.izz_s + self.inn_s

        self.panel = panel
        self._eval_matrices()

    def _eval_matrices(self):
        """Compute stiffness and mass matrices using Gauss-Legendre."""
        # Set up Gauss-Legendre quadrature points and weights
        xi_g, wi_g = np.polynomial.legendre.leggauss(len(self.panel.xi))
        self.xi_stf = xi_g
        self.wi_stf = wi_g

        # Map Gauss points to local and global coordinates
        x_local = (self.xi_stf + 1) * self.length / 2
        self.y_stf = self.y0 + x_local * np.sin(self.theta)
        self.x_stf = self.x0 + x_local * np.cos(self.theta)

        # Transformation to panel skewed/curved coordinate system
        ybar_stf = self.y_stf / np.cos(self.panel.beta)
        xbar_stf = self.x_stf - ybar_stf * np.sin(self.panel.beta)

        # Panel nondimensional coordinates (-1 to 1)
        xi_panel = 2 * xbar_stf / self.panel.a - 1
        eta_panel = 2 * ybar_stf / self.panel.b - 1

        # Mesh for integration: [xi, eta, id_xi, id_eta, weight]
        self.mesh = np.zeros((len(self.xi_stf), 5))
        for kee, ee in enumerate(xi_panel):
            self.mesh[kee, :] = [ee, eta_panel[kee], kee, kee, wi_g[kee]]

        xys = np.array(self.mesh[:, [0, 1]])

        # Helper to fetch and derive functions
        def get_basis(base_set, attr1, attr2):
            return Function.evaluate_basis_product(
                base_set, xys, self.panel.basis_type, attr1, attr2
            )

        # Evaluate basis functions and derivatives
        nu = get_basis(self.panel.base_u, "vfunc", "vfunc")
        nu_e = get_basis(self.panel.base_u, "vdfunc", "vfunc")
        nu_n = get_basis(self.panel.base_u, "vfunc", "vdfunc")

        nv = get_basis(self.panel.base_v, "vfunc", "vfunc")
        nv_e = get_basis(self.panel.base_v, "vdfunc", "vfunc")
        nv_n = get_basis(self.panel.base_v, "vfunc", "vdfunc")

        nw = get_basis(self.panel.base_w, "vfunc", "vfunc")
        nw_e = get_basis(self.panel.base_w, "vdfunc", "vfunc")
        nw_n = get_basis(self.panel.base_w, "vfunc", "vdfunc")
        nw_ee = get_basis(self.panel.base_w, "vd2func", "vfunc")
        nw_nn = get_basis(self.panel.base_w, "vfunc", "vd2func")
        nw_en = get_basis(self.panel.base_w, "vdfunc", "vdfunc")

        nbx = get_basis(self.panel.base_bx, "vfunc", "vfunc")
        nbx_e = get_basis(self.panel.base_bx, "vdfunc", "vfunc")
        nbx_n = get_basis(self.panel.base_bx, "vfunc", "vdfunc")

        nby = get_basis(self.panel.base_by, "vfunc", "vfunc")
        nby_e = get_basis(self.panel.base_by, "vdfunc", "vfunc")
        nby_n = get_basis(self.panel.base_by, "vfunc", "vdfunc")

        zeros = np.zeros_like(nu)
        # Compatibility with panel theory (only u, v and w for Kirchhoff)
        if self.panel.theory == StructuralTheory.KIRCHHOFF:
            nu_ee = get_basis(self.panel.base_u, "vd2func", "vfunc")
            nu_nn = get_basis(self.panel.base_u, "vfunc", "vd2func")
            nu_en = get_basis(self.panel.base_u, "vdfunc", "vdfunc")

            nv_ee = get_basis(self.panel.base_v, "vd2func", "vfunc")
            nv_nn = get_basis(self.panel.base_v, "vfunc", "vd2func")
            nv_en = get_basis(self.panel.base_v, "vdfunc", "vdfunc")

            nw_eee = get_basis(self.panel.base_w, "vd3func", "vfunc")
            nw_nnn = get_basis(self.panel.base_w, "vfunc", "vd3func")
            nw_een = get_basis(self.panel.base_w, "vd2func", "vdfunc")
            nw_enn = get_basis(self.panel.base_w, "vdfunc", "vd2func")

            nbx, nby = -nw_e, -nw_n

            nbx_e, nby_e = -nw_ee, -nw_en
            nbx_n, nby_n = -nw_en, -nw_nn

            nbx_ee, nby_ee = -nw_eee, -nw_een
            nbx_nn, nby_nn = -nw_enn, -nw_nnn
            nbx_en, nby_en = -nw_een, -nw_enn

            # Second order derivatives for Kirchhoff-Love theory
            b1_ee = np.block(
                [
                    [nu_ee, zeros, zeros, zeros, zeros],
                    [zeros, nv_ee, zeros, zeros, zeros],
                    [zeros, zeros, nw_ee, zeros, zeros],
                    [zeros, zeros, zeros, nbx_ee, zeros],
                    [zeros, zeros, zeros, zeros, nby_ee],
                ]
            )

            b1_nn = np.block(
                [
                    [nu_nn, zeros, zeros, zeros, zeros],
                    [zeros, nv_nn, zeros, zeros, zeros],
                    [zeros, zeros, nw_nn, zeros, zeros],
                    [zeros, zeros, zeros, nbx_nn, zeros],
                    [zeros, zeros, zeros, zeros, nby_nn],
                ]
            )

            b1_en = np.block(
                [
                    [nu_en, zeros, zeros, zeros, zeros],
                    [zeros, nv_en, zeros, zeros, zeros],
                    [zeros, zeros, nw_en, zeros, zeros],
                    [zeros, zeros, zeros, nbx_en, zeros],
                    [zeros, zeros, zeros, zeros, nby_en],
                ]
            )

        # Construct block matrices for displacement and derivatives
        b1 = np.block(
            [
                [nu, zeros, zeros, zeros, zeros],
                [zeros, nv, zeros, zeros, zeros],
                [zeros, zeros, nw, zeros, zeros],
                [zeros, zeros, zeros, nbx, zeros],
                [zeros, zeros, zeros, zeros, nby],
            ]
        )

        b1_e = np.block(
            [
                [nu_e, zeros, zeros, zeros, zeros],
                [zeros, nv_e, zeros, zeros, zeros],
                [zeros, zeros, nw_e, zeros, zeros],
                [zeros, zeros, zeros, nbx_e, zeros],
                [zeros, zeros, zeros, zeros, nby_e],
            ]
        )

        b1_n = np.block(
            [
                [nu_n, zeros, zeros, zeros, zeros],
                [zeros, nv_n, zeros, zeros, zeros],
                [zeros, zeros, nw_n, zeros, zeros],
                [zeros, zeros, zeros, nbx_n, zeros],
                [zeros, zeros, zeros, zeros, nby_n],
            ]
        )

        # Second order derivatives not required for Reissner-Mindlin theory
        if self.panel.theory == StructuralTheory.REISSNER_MINDLIN:
            b1_ee = b1_nn = b1_en = np.zeros_like(b1)

        # Mapping derivatives from panel (xi, eta) to (x, y)
        dedx = 2 / self.panel.a
        dedy = -2 / self.panel.a * np.tan(self.panel.beta)
        dndy = 2 / self.panel.b / np.cos(self.panel.beta)

        # Directional cosines for stiffener orientation
        dxdt, dydt = np.cos(self.theta), np.sin(self.theta)
        dxdn, dydn = -np.sin(self.theta), np.cos(self.theta)

        # Reassemble to ease compatibility in case of Kirchhoff-Love theory
        t_skew = np.array(
            [
                [1, np.sin(self.panel.beta), 0, 0, 0],
                [0, np.cos(self.panel.beta), 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, np.cos(self.panel.beta), 0],
                [0, 0, 0, -np.sin(self.panel.beta), 1],
            ]
        )
        c, s = np.cos(self.theta), np.sin(self.theta)
        z = self.eccentricity

        # Compatibility with panel displacements
        self.t_st = np.array(
            [
                [c, s, 0, z * c, z * s],
                [-s, c, 0, -z * s, z * c],
                [0, 0, 1, 0, 0],
                [0, 0, 0, c, s],
                [0, 0, 0, -s, c],
            ]
        )

        # Compute displacements and derivatives in panel coordinates
        up = t_skew @ b1
        up_x = t_skew @ b1_e * dedx
        up_y = t_skew @ (b1_e * dedy + b1_n * dndy)
        up_xx = t_skew @ b1_ee * dedx**2
        up_yy = t_skew @ (
            (b1_ee * dedy + b1_en * dndy) * dedy
            + (b1_en * dedy + b1_nn * dndy) * dndy
        )
        up_xy = t_skew @ (b1_ee * dedx + b1_en * dedy) * dedx

        # Transformation to stiffener local system (t, n, z)
        self.ust = self.t_st @ up
        self.ust_t = self.t_st @ (up_x * dxdt + up_y * dydt)
        self.ust_n = self.t_st @ (up_x * dxdn + up_y * dydn)
        self.ust_tt = self.t_st @ (
            (up_xx * dxdt + up_xy * dydt) * dxdt
            + (up_xy * dxdt + up_yy * dydt) * dydt
        )
        self.ust_nn = self.t_st @ (
            (up_xx * dxdn + up_xy * dydn) * dxdn
            + (up_xy * dxdn + up_yy * dydn) * dydn
        )
        self.ust_nt = self.t_st @ (
            (up_xx * dxdn + up_xy * dydn) * dxdt
            + (up_xy * dxdn + up_yy * dydn) * dydt
        )

        isotropic = hasattr(self.laminate.plies[0].material, "E")

        def mat_select(ltot, ctot, lines, cols):
            """Auxiliary matrix for Degree of Freedom (DOF) selection."""
            ans = np.zeros((ltot, ctot))
            for line, c in zip(lines, cols):
                ans[int(line), int(c)] = 1
            return ans

        # Constitutive matrix and DOF mapping for Timoshenko Beam
        if self.theory == StructuralTheory.TIMOSHENKO:
            kc = 5 / 6  # Shear correction factor
            if isotropic:
                e = self.laminate.plies[0].material.E
                nu = self.laminate.plies[0].material.nu
                g = e / (2 * (1 + nu))
                k11, k22 = e * self.area, kc * g * self.area
                k33, k44 = kc * g * self.area, kc * g * self.area
                k55, k66 = e * self.inn_s, g * self.j0_s

                self.c0 = np.diag([k11, k22, k33, k44, k55, k66])
                # Note: manual matrix assembly for k33/k44 coupling if needed
                self.c0[2, 3] = self.c0[3, 2] = k33

                self._TKu = mat_select(6, 5, [3], [3])
                self._TKu_t = mat_select(
                    6, 5, [0, 1, 2, 4, 5], [0, 1, 2, 3, 4]
                )
                self._TKu_tt = mat_select(6, 5, [], [])
                self._TKu_nt = mat_select(6, 5, [], [])
            else:
                # Composite Reduced Stiffness
                h = self.height
                ar = self.laminate.A_red
                br = self.laminate.B_red
                dr = self.laminate.D_red

                k11, k33 = h * ar[0, 0], h * ar[1, 1]
                k44, k55 = h * ar[1, 1], (h**3 / 12) * ar[0, 0]
                k66 = h * dr[1, 1] / 3
                k16, b16, b66 = h * ar[0, 1], -h * br[0, 1], -h * br[1, 1]

                self.c0 = np.array(
                    [
                        [k11, k16, k16, 0, b16],
                        [k16, k33, k33, 0, b66],
                        [k16, k33, k44, 0, b66],
                        [0, 0, 0, k55, 0],
                        [b16, b66, b66, 0, k66],
                    ]
                )

                self._TKu = mat_select(5, 5, [2], [3])
                self._TKu_t = mat_select(5, 5, [0, 1, 3, 4], [0, 2, 3, 4])
                self._TKu_tt = mat_select(5, 5, [], [])
                self._TKu_nt = mat_select(5, 5, [], [])

            # Mass matrix components
            rho = self.laminate.plies[0].material.rho
            m11 = m22 = m33 = rho * self.area
            m44, m55 = rho * self.inn_s, rho * self.j0_s
            self.i0 = np.diag([m11, m22, m33, m44, m55])

            self._TMu = mat_select(5, 5, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
            self._TMu_t = mat_select(5, 5, [], [])
            self._TMu_n = mat_select(5, 5, [], [])

        # Constitutive matrix for Euler-Bernoulli Beam
        elif self.theory == StructuralTheory.EULER_BERNOULLI:
            if isotropic:
                e = self.laminate.plies[0].material.E
                nu = self.laminate.plies[0].material.nu
                g = e / (2 * (1 + nu))
                k11, k22 = e * self.area, e * self.izz_s
                k33, k44 = e * self.inn_s, g * self.j0_s

                self.c0 = np.diag([k11, k22, k33, k44])
                self._TKu = mat_select(4, 5, [], [])
                self._TKu_t = mat_select(4, 5, [0], [0])
                self._TKu_tt = mat_select(4, 5, [1, 2], [1, 2])
                self._TKu_nt = mat_select(4, 5, [3], [2])

            rho = self.laminate.plies[0].material.rho
            m11 = m22 = m33 = rho * self.area
            m44, m55 = rho * self.inn_s, rho * self.j0_s
            self.i0 = np.diag([m11, m22, m33, m44, m55])

            self._TMu = mat_select(5, 5, [0, 1, 2], [0, 1, 2])
            self._TMu_t = mat_select(5, 5, [3], [2])
            self._TMu_n = mat_select(5, 5, [4], [2])

        # Integrate and assemble global matrices
        aux_k = (
            self._TKu @ self.ust
            + self._TKu_t @ self.ust_t
            + self._TKu_nt @ self.ust_nt
            + self._TKu_tt @ self.ust_tt
        )
        aux_kt = np.transpose(aux_k, [0, 2, 1])

        aux_m = (
            self._TMu @ self.ust
            + self._TMu_t @ self.ust_t
            + self._TMu_n @ self.ust_n
        )
        aux_mt = np.transpose(aux_m, [0, 2, 1])

        self._K = (
            np.tensordot(self.wi_stf, aux_kt @ self.c0 @ aux_k, [0, 0])
            * self.jac
        )
        self._M = (
            np.tensordot(self.wi_stf, aux_mt @ self.i0 @ aux_m, [0, 0])
            * self.jac
        )

        len_u, len_v, len_w = (
            len(self.base_u),
            len(self.base_v),
            len(self.base_w),
        )

        # Compatibility with panel theory (only u, v and w for Kirchhoff)
        if self.panel.theory == StructuralTheory.KIRCHHOFF:
            self._M = self._M[:len_u + len_v + len_w, :len_u + len_v + len_w]
            self._K = self._K[:len_u + len_v + len_w, :len_u + len_v + len_w]

    def init_nlin(self, panel):
        """Initialize nonlinear matrix components for large deflections.

        Args:
            panel (Panel): The parent panel object.
        """
        lenw = len(panel.base_w)
        lenu, lenv = len(panel.base_u), len(panel.base_v)
        lenbx, lenby = len(panel.base_bx), len(panel.base_by)
        len_xi = len(self.xi_stf)

        # Reduced stiffener coefficients for nonlinear terms
        a11 = panel.mat.Cst_RM[0, 0]
        a16 = panel.mat.Cst_RM[0, 1]
        b16 = panel.mat.Cst_RM[0, 4]

        aa11 = np.array([[a11]])
        cnl = 2 * np.diag([a11, a16, a16, b16])

        # Slice basis derivatives for nonlinear kinematic relations
        nu_t = self.ust_t[:, [0], :lenu]
        nw_t = self.ust_t[:, [2], lenu + lenv : lenu + lenv + lenw]
        nbt_t = self.ust_t[
            :, [3], lenu + lenv + lenw : lenu + lenv + lenw + lenbx
        ]
        nbn_t = self.ust_t[:, [4], lenu + lenv + lenw + lenbx :]

        wgs = self.mesh[:, -1]

        # Build gradient matrix for nonlinear strains
        n_bar_w_t = np.zeros((len_xi, lenw, int((lenw + 1) * lenw / 2)))
        for km in range(len_xi):
            n_bar_w_t[km, :, :lenw] = np.diag(nw_t[km, 0, :])
            for kn in range(lenw):
                start = int((1 + kn) * lenw - (1 + kn) * kn / 2)
                end = int((2 + kn) * lenw - (1 + kn) * (2 + kn) / 2)
                n_bar_w_t[km, kn, start:end] = nw_t[km, 0, kn + 1 :]
                n_bar_w_t[km, kn + 1 :, start:end] = nw_t[km, 0, kn] * np.eye(
                    int(lenw - (1 + kn))
                )

        n_bar = 0.5 * nw_t @ n_bar_w_t
        dn_bar = nw_t @ n_bar_w_t
        dn_bar_t = np.transpose(dn_bar, [0, 2, 1])

        zeros = np.zeros_like(nw_t)
        n_bar2 = np.tile(n_bar, (4, 1, 1))
        dn_bar2 = np.tile(dn_bar, (4, 1, 1))
        dn_bar2_t = np.transpose(dn_bar2, [0, 2, 1])

        unl = np.block(
            [
                [nu_t, zeros, zeros, zeros, zeros],
                [zeros, zeros, nw_t, zeros, zeros],
                [zeros, zeros, zeros, nbt_t, zeros],
                [zeros, zeros, zeros, zeros, nbn_t],
            ]
        )
        unlt = np.transpose(unl, [0, 2, 1])

        # Integrate nonlinear contributions
        self.nlnl = (
            np.tensordot(wgs, dn_bar_t @ aa11 @ n_bar, [0, 0]) * self.jac
        )
        self.unl = np.tensordot(wgs, unlt @ cnl @ n_bar2, [0, 0]) * self.jac
        self.nlu = np.tensordot(wgs, dn_bar2_t @ cnl @ unl, [0, 0]) * self.jac

        self.lenw, self.lenu, self.lenv = lenw, lenu, lenv
        self.lenbx, self.lenby = lenbx, lenby

    def compute_nl(self, ut):
        """Compute the nonlinear force vector for a given displacement.

        Args:
            ut (ndarray): Global displacement vector.

        Returns:
            ndarray: Nonlinear internal force vector.
        """
        lw, lu, lv = self.lenw, self.lenu, self.lenv
        lbx, lby = self.lenbx, self.lenby
        total_dof = lu + lv + lw + lbx + lby

        # Extract transverse displacement DOFs
        u3 = ut[lu + lv : lu + lv + lw]
        nu3_bar = np.zeros((int((lw + 1) * lw / 2), lw))

        nu3_bar[:lw, :] = np.diag(u3[:, 0])
        for kk, uu in enumerate(u3):
            start = int((1 + kk) * lw - (1 + kk) * kk / 2)
            end = int((2 + kk) * lw - (1 + kk) * (2 + kk) / 2)
            nu3_bar[start:end, kk] = 0.5 * u3[kk + 1 :, 0]
            nu3_bar[start:end, kk + 1 :] = (
                0.5 * uu * np.eye(int(lw - (1 + kk)))
            )

        nu3q = np.zeros((int((lw + 1) * lw / 2), total_dof))
        nu3q[:, lu + lv : lu + lv + lw] = nu3_bar

        # Nonlinear force terms
        fnl = nu3q.T @ self.nlnl @ nu3q @ ut
        f_acp = self.unl @ nu3q @ ut
        f_acp += nu3q.T @ self.nlu @ ut

        return fnl + f_acp
