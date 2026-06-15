"""Verification for comparing results with Dixon and Mei (1993).

This script benchmark-test post flutter limit cycle oscillation (LCO)
amplitudes of a simply supported isotropic square panel against literature
results of Dixon and Mei (1993) available at https://doi.org/10.2514/3.11606
and Tsunematsu et al (2021) available at
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
from openpanelflutter.material import Laminate, Orthotropic
from openpanelflutter.panel import Panel

# Configuring the global Matplotlib style
apply_plot_style()

CM_TO_INCH = 0.393701
WIDTH = 13.0
FIGWIDTH = WIDTH * CM_TO_INCH
FIG_RATIO = 0.65
FIGHEIGHT = FIGWIDTH * FIG_RATIO

CURRENT_DIR = Path(__file__).resolve().parent

# --- Reference Database from Dixon and Mei (1993) ---
DIXON_MEI_DATABASE = {
    165: 0.0,
    195: 0.5104,
    225: 0.7195,
    255: 0.8779,
    285: 0.9971,
}

# --- Reference Database from Tsunematsu et al (2021) ---
TSUNEMATSU_DATABASE = {
    165: 0.0,
    195: 0.5104,
    225: 0.7195,
    255: 0.8583,
    285: 1.0167,
}

# --- Reference material properties and laminate ---
AS4_PEEK = Orthotropic(
    "AS4/PEEK",
    rho=1600,
    E1=207e9,
    E2=5.17e9,
    G12=2.59e9,
    G13=2.59e9,
    G23=2.59e9,
    nu12=0.25,
)
LAYUP_ANGLES = [0, 90, 0, 90, 0, 90]
LAMINATE = Laminate("Laminate")
LAMINATE.add_stack(
    AS4_PEEK, thicknesses=3.05e-3 / len(LAYUP_ANGLES), angles=LAYUP_ANGLES
)


def print_terminal_summary_critical(lambda_cr):
    """Print an verification matrix block directly to the stdout terminal."""
    # Extract references for Table 1 (First key represents lambda_crit)
    ref_lambda_dm = list(DIXON_MEI_DATABASE.keys())[0]
    ref_lambda_ts = list(TSUNEMATSU_DATABASE.keys())[0]

    # Calculate errors for Table 1
    err_abs_dm = lambda_cr - ref_lambda_dm
    err_rel_dm = (err_abs_dm / ref_lambda_dm) * 100

    err_abs_ts = lambda_cr - ref_lambda_ts
    err_rel_ts = (err_abs_ts / ref_lambda_ts) * 100

    print("\n" + "=" * 94)
    print(
        " TABLE 1: CRITICAL AERODYNAMIC PRESSURE PARAMETER (λ_crit) COMPARISON"
    )
    print("=" * 94)
    print(
        f"{'Source':<25} | {'λ_crit':<10} | {'Abs. Error':<12} |"
        f" {'Rel. Error (%)':<15}"
    )
    print("-" * 94)
    print(
        f"{'Dixon and Mei (1993)':<25} |"
        f" {ref_lambda_dm:<10.2f} | {'-':<12} | {'-':<15}"
    )
    print(
        f"{'Tsunematsu et al. (2021)':<25} |"
        f" {ref_lambda_ts:<10.2f} | {'-':<12} | {'-':<15}"
    )
    print("-" * 94)
    print(
        f"{'Present Work':<25} | {lambda_cr:<10.2f} |"
        f" {err_abs_dm:<+12.2f} | {err_rel_dm:<+15.2f} (vs Dixon and Mei)"
    )
    print(
        f"{'':<25} | {'':<10} | {err_abs_ts:<+12.2f} |"
        f" {err_rel_ts:<+15.2f} (vs Tsunematsu et al.)"
    )
    print("=" * 94 + "\n")


def print_terminal_summary_amplitude(my_amplitudes):
    """Print a verification matrix block directly to the stdout terminal."""
    # =========================================================================
    # TABLE 2: NON-DIMENSIONAL AMPLITUDE COMPARISON (Values > 0)
    # =========================================================================
    print("")
    print("=" * 108)
    print(" NON-DIMENSIONAL LCO AMPLITUDE COMPARISON")
    print("=" * 108)
    # Present Work || Dixon & Mei Block || Tsunematsu Block
    header = (
        f"{'λ':<6} | {'Present Work':<13} || "
        f"{'Dixon and Mei':<14} | {'Abs. Err.':<10} | {'% Err.':<8} || "
        f"{'Tsunematsu et al.':<18} | {'Abs. Err.':<10} | {'% Err.':<8}"
    )
    print(header)
    print("-" * 108)

    # Filter keys where reference values are strictly greater than zero
    active_lambdas = [k for k, v in DIXON_MEI_DATABASE.items() if v > 0.0]

    for kl, lam in enumerate(active_lambdas):
        val_dm = DIXON_MEI_DATABASE[lam]
        val_ts = TSUNEMATSU_DATABASE[lam]
        val_my = my_amplitudes[kl]

        # Compute amplitude errors for Dixon and Mei
        amp_abs_err_dm = val_my - val_dm
        amp_rel_err_dm = (amp_abs_err_dm / val_dm) * 100 if val_dm > 0 else 0.0

        # Compute amplitude errors for Tsunematsu et al.
        amp_abs_err_ts = val_my - val_ts
        amp_rel_err_ts = (amp_abs_err_ts / val_ts) * 100 if val_ts > 0 else 0.0

        # Row print aligned dynamically with the specified header boundaries
        print(
            f"{lam:<6} | {val_my:<13.4f} || "
            f"{val_dm:<14.4f} | {amp_abs_err_dm:<+10.4f} |"
            f" {amp_rel_err_dm:<+8.2f} || "
            f"{val_ts:<18.4f} | {amp_abs_err_ts:<+10.4f} |"
            f" {amp_rel_err_ts:<+8.2f}"
        )
    print("=" * 108)
    print("")


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
    # Panel setup: a=0.305, b=0.305
    panel = Panel(0.305, 0.305, LAMINATE)
    panel.setup_kinematics(
        n_func,
        theory=theory,
        basis_type=basis_type,
        boundary_conditions=boundary_conditions,
        n_gauss=n_gauss,
    )
    analyses = Analysis(panel)
    print("Flutter analysis: ", end="")
    analyses.run_lambda_sweep(lambda_min=160, lambda_max=165, n_points=200)
    analyses.identify_flutter()
    print_terminal_summary_critical(analyses.lamb_cr_interp)

    max_amp, local_amp = analyses.post_flutter_sim(
        lambdas, t_end, delta_t, -1, parallel=parallel, num_proc=4
    )

    print_terminal_summary_amplitude(local_amp)

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
        list(DIXON_MEI_DATABASE.keys()),
        list(DIXON_MEI_DATABASE.values()),
        "--sy",
        markersize=3.5,
        label="Dixon and Mei (1993)",
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
    plt.xlim([160, 290])
    plt.ylim([-0.05, 1.05])
    if save_figures:
        output_dir = CURRENT_DIR / "figures" / "compare_Dixon_Mei"
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            str(output_dir / "limit_cycle_DixonMei.pdf"), bbox_inches="tight"
        )
    else:
        plt.show()


if __name__ == "__main__":
    # Execute benchmark case
    run_simulations(
        n_func=6,
        lambdas=[
            key for key, value in DIXON_MEI_DATABASE.items() if value > 0.0
        ],
        delta_t=1e-6,
        t_end=0.1,
        parallel=True,
        save_figures=True,
    )
