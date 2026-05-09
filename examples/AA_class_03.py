# %%
"""AA-250: Aeroelasticity Comprehensive Examples - Slide 02.

This script contains a suite of examples for validating panel flutter,
thermal buckling, and vibration modes of curved/flat aerospace panels.
All examples follow the AE-250 Course curriculum (ITA).
"""

import matplotlib.pyplot as plt
import numpy as np

from Fcns.analysis import Analysis
from Fcns.definitions import BasisFunction, StructuralTheory, apply_plot_style
from Fcns.material import Isotropic, Laminate
from Fcns.panel import Panel

# Configuring the global Matplotlib style
apply_plot_style()

CM_TO_INCH = 0.393701
WIDTH = 13.0
FIGWIDTH = WIDTH * CM_TO_INCH
FIG_RATIO = 0.65
FIGHEIGHT = FIGWIDTH * FIG_RATIO


def get_default_laminate():
    """Create a standard Aluminum 2024-T3 laminate for Dowell example."""
    al_2024_t3 = Isotropic("AL2024T3", 2700, 70e9, 0.33)
    laminate = Laminate("Aluminum_Skin")
    laminate.add_stack(al_2024_t3, 2e-3)
    return laminate


def run_flutter_analysis(n_func, lamb):
    """Perform flutter analysis for a given panel curvature.

    Args:
        n_func (int): Number of basis functions used in the panel kinematics.
        lamb (array_like): Aerodynamic loading parameters for the flutter scan.
    """
    laminate = get_default_laminate()

    # Panel setup: a=0.3, b=0.3
    panel = Panel(0.3, 0.3, laminate)
    panel.setup_kinematics(
        n_func,
        theory=StructuralTheory.KIRCHHOFF,
        basis_type=BasisFunction.SINES,
        n_gauss=30,
    )

    sim = Analysis(panel)

    wa_max, wa_point = sim.post_flutter_sim(
        lamb, 0.2, 1e-7, -1, parallel=True, num_proc=4
    )

    for kr, result in enumerate(sim.results):
        h = sim.panel.laminate.total_thickness
        ws = (
            sim.panel.get_displacement_at_points(
                np.array([0.5, 0.0]), 2, result.displ
            )
            / h
        )
        time = result.time_array

        plt.figure(kr, figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
        plt.plot(time, ws, "b")
        plt.gca().grid(
            visible=True, which="both", linestyle=":", linewidth=0.5
        )
        plt.ylabel("$w$ $[-]$", fontsize=10)
        plt.xlabel("Time $[s]$", fontsize=10)
        plt.show()

    return wa_max, wa_point


if __name__ == "__main__":
    print("--- Starting AE-250 Examples ---\n")

    # Reference Datas
    wa_dowell = np.array([0, 0.73, 1.00, 1.18, 1.33])
    wa_tsunematsu = np.array([0, 0.69, 0.97, 1.17, 1.27])
    lambs_ref = np.array([515, 635, 756, 876, 993])
    lambs = np.array([635, 756, 876, 993])
    n_func = 2
    wa, wa_p = run_flutter_analysis(n_func, lambs)

    plt.figure(5, figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
    plt.plot(
        lambs_ref,
        wa_tsunematsu,
        "--^b",
        markersize=3.5,
        label=r"Tsunematsu \it{et al.}",
    )
    plt.plot(lambs_ref, wa_dowell, "--sy", markersize=3.5, label="Dowell")
    plt.plot(lambs, wa_p, "--or", markersize=3.5, label="Present Work")
    plt.legend(
        loc="best",
        facecolor="white",
        framealpha=1,
        handlelength=1.7,
        markerscale=1,
    )
    plt.gca().grid(visible=True, which="both", linestyle=":", linewidth=0.5)
    plt.ylabel("$w_A$ $[-]$", fontsize=10)
    plt.xlabel(r"$\lambda$ $[-]$", fontsize=10)
    plt.xlim([500, 1050])
    # plt.ylim([-0.05,1.4])
    plt.show()

# %%
