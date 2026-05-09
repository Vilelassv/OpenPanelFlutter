"""Material and Laminate properties for aerospace structural analysis.

This module implements Classical Laminate Theory (CLT) to compute
stiffness matrices (ABD) for composite shells and stiffeners.
"""

import numpy as np

from .definitions import MaterialType


class Material:
    """Base class for all materials."""

    def __init__(self, name: str, rho: float):
        """Initialize the Material.

        Args:
            name (str): Identification name of the material.
            rho (float): Mass density in [kg/m³].
        """
        self.name = name
        self.rho = rho  # Density [kg/m3]


class Isotropic(Material):
    """Isotropic material with constant properties in all directions."""

    def __init__(self, name, rho, E, nu, alpha=0, damping=0):
        """Initialize an Isotropic material.

        Args:
            name (str): Material identification.
            rho (float): Density [kg/m³].
            E (float): Young's Modulus [Pa].
            nu (float): Poisson's ratio [-].
            alpha (float): Thermal expansion coefficient [1/°C].
            damping (float): Material damping ratio [%].
        """
        super().__init__(name, rho)
        self.kind = MaterialType.ISOTROPIC
        self.E = E  # Young's Modulus [Pa]
        self.nu = nu  # Poisson's ratio [-]
        self.alpha = alpha  # Thermal expansion coefficient [1/oC]
        self.damping = damping  # Material damping ratio [%]


class Orthotropic(Material):
    """Orthotropic material with distinct properties in orthogonal axes."""

    def __init__(
        self, name, rho, E1, E2, G12, G13, G23, nu12, alpha1=0, alpha2=0
    ):
        """Initialize an Orthotropic material (Ply level).

        Args:
            name (str): Material identification.
            rho (float): Density [kg/m³].
            E1 (float): Young's Modulus in fiber direction [Pa].
            E2 (float): Young's Modulus transverse to fiber [Pa].
            G12 (float): In-plane shear modulus [Pa].
            G13 (float): Out-of-plane shear modulus (1-3 plane) [Pa].
            G23 (float): Out-of-plane shear modulus (2-3 plane) [Pa].
            nu12 (float): Major Poisson's ratio [-].
            alpha1 (float): Thermal expansion in fiber direction [1/°C].
            alpha2 (float): Thermal expansion transverse to fiber [1/°C].
        """
        super().__init__(name, rho)
        self.kind = MaterialType.ORTHOTROPIC
        self.E1 = E1
        self.E2 = E2
        self.nu12 = nu12
        self.nu21 = nu12 * (E2 / E1)
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23
        self.alpha1 = alpha1
        self.alpha2 = alpha2


class Laminate:
    """Collection of Plies forming a composite layup.

    Implements Classical Laminate Theory (CLT) calculations to obtain
    ABD matrices and equivalent properties.
    """

    class Ply:
        """Individual layer within a laminate."""

        def __init__(self, material, thickness, angle_deg=0):
            """Initialize a single Ply within a laminate.

            Args:
                material (Material): Material object.
                thickness (float): Thickness of the ply [m].
                angle_deg (float): Fiber orientation angle [degrees].
            """
            self.material = material
            self.h = thickness
            self.angle = angle_deg

    def __init__(self, name):
        """Initialize a Laminate layup container.

        Args:
            name (str): Identification name for the laminate sequence.
        """
        self.name = name
        self.plies = []

        # Initializing matrices to None
        self.A = None
        self.B = None
        self.D = None
        self.A_s = None
        self.A_red = None
        self.B_red = None
        self.D_red = None
        self.N_R = None
        self.M_R = None

    def add_stack(self, materials, thicknesses, angles=0):
        """Adds multiple plies to the laminate stack.

        Args:
            materials: Material object or list of objects.
            thicknesses: float or list of floats.
            angles: float or list of floats (default=0).
        """
        mats = [materials] if not isinstance(materials, list) else materials
        thks = (
            [thicknesses] if not isinstance(thicknesses, list) else thicknesses
        )
        angs = [angles] if not isinstance(angles, list) else angles

        num_layers = max(len(mats), len(thks), len(angs))

        if len(mats) == 1:
            mats *= num_layers
        if len(thks) == 1:
            thks *= num_layers
        if len(angs) == 1:
            angs *= num_layers

        if not (len(mats) == len(thks) == len(angs)):
            raise ValueError("Laminate error: Incompatible list lengths.")

        for m, t, a in zip(mats, thks, angs):
            self.plies.append(self.Ply(m, t, a))

    @property
    def total_thickness(self):
        """Returns the sum of all ply thicknesses."""
        return sum(p.h for p in self.plies)

    def compute_ABD(self):
        """Compute the ABD stiffness matrices and their reduced counterparts.

        Required for both shell and stiffener analysis.
        """
        total_h = self.total_thickness
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        A_s = np.zeros((2, 2))
        A_red = np.zeros((2, 2))
        B_red = np.zeros((2, 2))
        D_red = np.zeros((2, 2))
        # Thermal vectors
        N_R = np.zeros((3, 1))
        M_R = np.zeros((3, 1))

        for k, ply in enumerate(self.plies):
            ang = np.radians(ply.angle)
            mat = ply.material

            if mat.kind == MaterialType.ISOTROPIC:
                E1, E2, nu12, nu21 = mat.E, mat.E, mat.nu, mat.nu
                G12 = E1 / (2 * (1 + nu12))
                G23 = G13 = G12
                a1, a2 = mat.alpha, mat.alpha
            else:
                E1, E2, nu12, nu21 = mat.E1, mat.E2, mat.nu12, mat.nu21
                G12, G23, G13 = mat.G12, mat.G23, mat.G13
                a1, a2 = mat.alpha1, mat.alpha2
            # Vector alpha in material coordinates
            alpha_mat = np.array([[a1], [a2], [0]])

            # Constitutive matrices (Reduced Stiffness)
            q_div = 1 - nu12 * nu21
            Q = np.array(
                [
                    [E1 / q_div, nu12 * E2 / q_div, 0],
                    [nu12 * E2 / q_div, E2 / q_div, 0],
                    [0, 0, G12],
                ]
            )

            Qs = (5 / 6) * np.array([[G23, 0], [0, G13]])

            # Transformation matrices
            c, s = np.cos(ang), np.sin(ang)
            T_b = np.array(
                [
                    [c**2, s**2, c * s],
                    [s**2, c**2, -c * s],
                    [-2 * c * s, 2 * c * s, c**2 - s**2],
                ]
            )

            # To transform alpha from material to global:
            # alpha_global = inv(T_b) @ alpha_mat
            c_i, s_i = np.cos(-ang), np.sin(-ang)
            T_b_inv = np.array(
                [
                    [c_i**2, s_i**2, c_i * s_i],
                    [s_i**2, c_i**2, -c_i * s_i],
                    [-2 * c_i * s_i, 2 * c_i * s_i, c_i**2 - s_i**2],
                ]
            )

            alpha_global = T_b_inv @ alpha_mat

            T_s = np.array([[c, -s], [s, c]])

            Q_bar = T_b.T @ Q @ T_b
            Qs_bar = T_s.T @ Qs @ T_s

            # Z-coordinates for integration
            nk = (k * ply.h) - total_h / 2
            nk1 = ((k + 1) * ply.h) - total_h / 2
            nbk = (k + 0.5) * ply.h - total_h / 2

            A += Q_bar * ply.h
            B += Q_bar * ply.h * nbk
            D += Q_bar * (nk1**3 - nk**3) / 3
            A_s += Qs_bar * ply.h

            # Thermal load integration
            N_R += Q_bar @ alpha_global * ply.h
            M_R += Q_bar @ alpha_global * ply.h * nbk

            # Reduced stiffness for stiffeners (1D behavior)
            q11_r = Q_bar[0, 0] - (Q_bar[0, 1] ** 2 / Q_bar[1, 1])
            q16_r = Q_bar[0, 2] - (Q_bar[0, 1] * Q_bar[1, 2] / Q_bar[1, 1])
            q66_r = Q_bar[2, 2] - (Q_bar[0, 1] ** 2 / Q_bar[1, 1])
            Q_red = np.array([[q11_r, q16_r], [q16_r, q66_r]])

            A_red += Q_red * ply.h
            B_red += Q_red * ply.h * nbk
            D_red += Q_red * (nk1**3 - nk**3) / 3

        # Clean-up near-zero values
        for mat in [A, B, D, A_s, A_red, B_red, D_red]:
            mat[np.abs(mat) < 1e-6] = 0

        # storage
        self.A, self.B, self.D, self.A_s = A, B, D, A_s
        self.A_red, self.B_red, self.D_red = A_red, B_red, D_red
        self.N_R, self.M_R = N_R, M_R
