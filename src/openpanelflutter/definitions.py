"""Definitions and constants for the aeroelastic research package.

This module centralizes the enums and global constants used across
the panel and stiffener modules to ensure consistency and avoid
circular imports.
"""

import shutil
from enum import Enum

import matplotlib.pyplot as plt


class StructuralTheory(Enum):
    """Standard structural theories for plates and beams.

    This Enum maps the kinematic assumptions used for both the
    2D panel (Plate Theory) and the 1D stiffeners (Beam Theory).

    Attributes:
        KIRCHHOFF: Classical Plate Theory (CPT) - Panel.
                   Euler-Bernoulli - Stiffener.
        REISSNER_MINDLIN: First-order Shear Deformation Theory (FSDT) - Panel.
                          Timoshenko - Stiffener.
    """

    KIRCHHOFF = "KF"
    REISSNER_MINDLIN = "RM"
    EULER_BERNOULLI = "EB"
    TIMOSHENKO = "TM"


class MaterialType(Enum):
    """Classification of material constitutive behavior.

    Used to determine which mathematical model to apply for
    stiffness and failure analysis.
    """

    ISOTROPIC = "iso"
    ORTHOTROPIC = "ortho"


class BasisFunction(Enum):
    """Available basis functions for the Ritz method.

    Attributes:
        BARDELL: Hierarchical polynomials (Bardell, 1991).
        SINES: Standard sine trigonometric series.
        COSINES: Standard cosine trigonometric series.
    """

    BARDELL = "Bardell"
    SINES = "sines"
    COSINES = "cosines"


class BoundaryCondition(Enum):
    """Standard boundary conditions for the Ritz Method with Bardell.

    The four letters represent the edges:
    1: xi = -1 (left), 2: eta = -1 (bottom),
    3: xi = +1 (right), 4: eta = +1 (top).
    """

    SSSS = "SSSS"  # Hard Simply Supported on all edges
    SSSSsoft = "SSSSsoft"  # Soft Simply Supported on all edges
    CCCC = "CCCC"  # Clamped on all edges
    CFFF = "CFFF"
    CCSS = "CCSS"
    SFSF = "SFSF"
    CFCF = "CFCF"
    FCFF = "FCFF"


def apply_plot_style():
    """Configure the global Matplotlib style with automatic LaTeX detection.

    This function sets font families, math rendering engines, and axis
    formatting to ensure academic-standard visualizations. It attempts to
    detect a LaTeX installation (e.g., MikTeX or TeX Live) on the system.
    If found, it enables 'usetex'; otherwise, it falls back to the native
    'mathtext' engine with Computer Modern fonts to maintain consistency.
    """
    # Check if 'latex' executable exists in the system's PATH
    latex_exists = shutil.which("latex") is not None

    if latex_exists:
        # Enable external LaTeX rendering for maximum quality
        plt.rc("text", usetex=True)
    else:
        # Fallback to internal mathtext engine
        # 'cm' (Computer Modern) keeps the academic/LaTeX look for symbols
        plt.rc("text", usetex=False)
        plt.rcParams["mathtext.fontset"] = "cm"

    # Define font properties (independent of LaTeX availability)
    plt.rc("font", family="serif")
    plt.rc("font", weight="bold")

    # Global plot appearance settings (PEP 8 compliant spacing)
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["legend.fontsize"] = "12"
    plt.rcParams["axes.labelsize"] = "11"
    plt.rcParams["axes.titlesize"] = "12"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
