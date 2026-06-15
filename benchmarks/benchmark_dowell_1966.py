"""Verification for OpenPanelFlutter comparing results with Dowell (1966).

This script benchmark-test post flutter limit cycle oscillation (LCO)
amplitudes of a simply supported isotropic square panel against literature
results of Dowell (1966) available at https://doi.org/10.2514/3.3658 and
Tsunematsu et al (2021) available at
https://doi.org/https://doi.org/10.1016/j.tws.2021.107964.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openpanelflutter.analysis import Analysis
from openpanelflutter.definitions import (
    BasisFunction,
    BoundaryCondition,
    StructuralTheory,
    apply_plot_style,
)
from openpanelflutter.material import Isotropic, Laminate
from openpanelflutter.panel import Panel

# Configuring the global Matplotlib style
apply_plot_style()

CM_TO_INCH = 0.393701
WIDTH = 13.0
FIGWIDTH = WIDTH * CM_TO_INCH
FIG_RATIO = 0.65
FIGHEIGHT = FIGWIDTH * FIG_RATIO

CURRENT_DIR = Path(__file__).resolve().parent

# --- Reference Database from Dowell (1966) ---
DOWELL_DATABASE = {
    515: 0.0,
    635: 0.73,
    756: 1.00,
    876: 1.18,
    993: 1.33,
}

# --- Reference Database from Tsunematsu et al (2021) ---
TSUNEMATSU_DATABASE = {
    515: 0.0,
    635: 0.69,
    756: 0.97,
    876: 1.17,
    993: 1.27,
}

# --- Reference isotropic material properties and laminate ---
MAT_REF = Isotropic("MAT_REF", rho=2700, E=75e9, nu=0.33)
LAMINATE = Laminate("Laminate")
LAMINATE.add_stack(MAT_REF, thicknesses=2e-3)


def print_terminal_summary_critical(lambda_cr):
    """Print an verification matrix block directly to the stdout terminal."""
    # Extract references for Table 1 (First key represents lambda_crit)
    ref_lambda_dm = list(DOWELL_DATABASE.keys())[0]
    ref_lambda_ts = list(TSUNEMATSU_DATABASE.keys())[0]

    # Calculate errors for Table 1
    err_abs_dm = lambda_cr - ref_lambda_dm
    err_rel_dm = (err_abs_dm / ref_lambda_dm) * 100

    err_abs_ts = lambda_cr - ref_lambda_ts
    err_rel_ts = (err_abs_ts / ref_lambda_ts) * 100

    print("\n" + "=" * 80)
    print(
        " TABLE 1: CRITICAL AERODYNAMIC PRESSURE PARAMETER (λ_crit) COMPARISON"
    )
    print("=" * 80)
    print(
        f"{'Source':<25} | {'λ_crit':<10} | {'Abs. Error':<12} |"
        f" {'Rel. Error (%)':<15}"
    )
    print("-" * 80)
    print(
        f"{'Dixon & Mei (1993)':<25} |"
        f" {ref_lambda_dm:<10.2f} | {'-':<12} | {'-':<15}"
    )
    print(
        f"{'Tsunematsu et al. (2021)':<25} |"
        f" {ref_lambda_ts:<10.2f} | {'-':<12} | {'-':<15}"
    )
    print("-" * 80)
    print(
        f"{'Present Work':<25} | {lambda_cr:<10.2f} |"
        f" {err_abs_dm:<+12.2f} | {err_rel_dm:<+15.2f} (vs DM)"
    )
    print(
        f"{'':<25} | {'':<10} | {err_abs_ts:<+12.2f} |"
        f" {err_rel_ts:<+15.2f} (vs TS)"
    )
    print("=" * 80 + "\n")


def run_simulations(
    n_func,
    lambdas,
    delta_t,
    t_end,
    theory=StructuralTheory.REISSNER_MINDLIN,
    boundary_conditions=BoundaryCondition.SSSS,
    basis_type=BasisFunction.BARDELL,
    n_gauss=-1,
    parallel=True,
    num_proc=4,
    save_figures=False,
):
    """Run verification and generate benchmarking data against literature."""
    # Panel setup: a=0.3, b=0.3
    panel = Panel(0.3, 0.3, LAMINATE)
    panel.setup_kinematics(
        n_func,
        theory=theory,
        basis_type=basis_type,
        boundary_conditions=boundary_conditions,
        n_gauss=n_gauss,
    )
    analyses = Analysis(panel)
    print("Flutter analysis: ", end="")
    analyses.run_lambda_sweep(lambda_min=510, lambda_max=515, n_points=200)
    analyses.identify_flutter()
    print_terminal_summary_critical(analyses.lamb_cr_interp)

    max_amp, local_amp = analyses.post_flutter_sim(
        lambdas, t_end, delta_t, -1, parallel=parallel, num_proc=4
    )

    lambdas_sim = np.insert(lambdas, 0, analyses.lamb_cr_interp)
    local_amp_sim = np.insert(local_amp, 0, 0)

    plt.figure(1, figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
    plt.plot(
        list(TSUNEMATSU_DATABASE.keys()),
        list(TSUNEMATSU_DATABASE.values()),
        "--^b",
        markersize=3.5,
        label=r"Tsunematsu \it{et al.} (2021)",
    )
    plt.plot(
        list(DOWELL_DATABASE.keys()),
        list(DOWELL_DATABASE.values()),
        "--sy",
        markersize=3.5,
        label=r"Dowell (1966)",
    )
    plt.plot(
        lambdas_sim,
        local_amp_sim,
        "--or",
        markersize=3.5,
        label="Present Work",
    )
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
    plt.ylim([-0.05, 1.4])
    plt.show()


if __name__ == "__main__":
    # Execute benchmark case
    run_simulations(
        n_func=7,
        lambdas=[key for key, value in DOWELL_DATABASE.items() if value > 0.0],
        delta_t=1e-7,
        t_end=0.2,
        parallel=False,
    )
