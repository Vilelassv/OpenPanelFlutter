"""Verification for comparing results with Dixon and Mei (1993).

This script benchmark-test post flutter limit cycle oscillation (LCO)
amplitudes of a simply supported isotropic square panel against literature
results of Dixon and Mei (1993) available at https://doi.org/10.2514/3.11606
and Tsunematsu et al (2021) available at
https://doi.org/https://doi.org/10.1016/j.tws.2021.107964.
"""

import copy
import logging
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

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

logger = logging.getLogger("OpenPanelFlutter")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if the script is re-run in the same session
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

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


def print_simulation_setup(
    n_func,
    lambdas,
    delta_t,
    t_end,
    theory,
    boundary_conditions,
    basis_type,
    n_gauss,
    parallel,
    n_proc=None,
):
    """Print a summary of the simulation configuration."""
    logger.info("\n" + "=" * 80)
    logger.info(" SIMULATION RUNTIME SETUP & CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f" {'Structural Theory':<25} : {theory.name}")
    logger.info(f" {'Boundary Conditions':<25} : {boundary_conditions.name}")
    logger.info(f" {'Basis Function Type':<25} : {basis_type.name}")
    logger.info(f" {'Number of Functions':<25} : {n_func}")
    logger.info(f" {'Gauss Quadrature Points':<25} : {n_gauss}")
    logger.info("-" * 80)
    logger.info(f" {'Time Step (dt)':<25} : {delta_t:.1e} s")
    logger.info(f" {'Final Time (t_end)':<25} : {t_end:<5.2f} s")
    logger.info(f" {'Total Steps per Case':<25} : {int(t_end / delta_t):,}")
    logger.info(f" {'Cases (λ)':<25} : {len(lambdas)} cases {list(lambdas)}")
    logger.info("-" * 80)

    if parallel:
        logger.info(f" {'Execution Mode':<25} : PARALLEL BATCH")
        logger.info(f" {'Processors Allocated':<25} : {n_proc}")
    else:
        logger.info(f" {'Execution Mode':<25} : SERIAL BATCH")

    logger.info("=" * 80 + "\n")


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

    logger.info("\n" + "=" * 94)
    logger.info(" CRITICAL AERODYNAMIC PRESSURE PARAMETER (λ_crit) COMPARISON")
    logger.info("=" * 94)
    logger.info(
        f"{'Source':<25} | {'λ_crit':<10} | {'Abs. Error':<12} |"
        f" {'Rel. Error (%)':<15}"
    )
    logger.info("-" * 94)
    logger.info(
        f"{'Dixon and Mei (1993)':<25} |"
        f" {ref_lambda_dm:<10.2f} | {'-':<12} | {'-':<15}"
    )
    logger.info(
        f"{'Tsunematsu et al. (2021)':<25} |"
        f" {ref_lambda_ts:<10.2f} | {'-':<12} | {'-':<15}"
    )
    logger.info("-" * 94)
    logger.info(
        f"{'Present Work':<25} | {lambda_cr:<10.2f} |"
        f" {err_abs_dm:<+12.2f} | {err_rel_dm:<+15.2f} (vs Dixon and Mei)"
    )
    logger.info(
        f"{'':<25} | {'':<10} | {err_abs_ts:<+12.2f} |"
        f" {err_rel_ts:<+15.2f} (vs Tsunematsu et al.)"
    )
    logger.info("=" * 94 + "\n")


def print_terminal_summary_amplitude(my_amplitudes):
    """Print a verification matrix block directly to the stdout terminal."""
    logger.info("")
    logger.info("=" * 108)
    logger.info(" NON-DIMENSIONAL LCO AMPLITUDE COMPARISON")
    logger.info("=" * 108)
    # Present Work || Dixon & Mei Block || Tsunematsu Block
    header = (
        f"{'λ':<6} | {'Present Work':<13} || "
        f"{'Dixon and Mei':<14} | {'Abs. Err.':<10} | {'% Err.':<8} || "
        f"{'Tsunematsu et al.':<18} | {'Abs. Err.':<10} | {'% Err.':<8}"
    )
    logger.info(header)
    logger.info("-" * 108)

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
        logger.info(
            f"{lam:<6} | {val_my:<13.4f} || "
            f"{val_dm:<14.4f} | {amp_abs_err_dm:<+10.4f} |"
            f" {amp_rel_err_dm:<+8.2f} || "
            f"{val_ts:<18.4f} | {amp_abs_err_ts:<+10.4f} |"
            f" {amp_rel_err_ts:<+8.2f}"
        )
    logger.info("=" * 108)
    logger.info("")


def _single_simulation_worker(
    lamb_value,
    panel_template,
    delta_t,
    t_end,
):
    local_panel = copy.deepcopy(panel_template)
    analyses = Analysis(local_panel)

    return analyses.post_flutter_sim(lamb_value, t_end, delta_t, -1)


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
    save_animation=False,
    show_arrow=False,
):
    """Run verification and generate benchmarking data against literature."""
    subcase = f"theory_{theory.value}_func_{basis_type.value}_nfunc_{n_func}"
    output_dir = CURRENT_DIR / "figures" / "compare_Dixon_Mei" / subcase
    output_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = output_dir / "benchmark_run.log"
    file_handler = logging.FileHandler(
        log_filepath, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    # Panel setup: a=0.305, b=0.305
    panel = Panel(0.305, 0.305, LAMINATE)
    panel.setup_kinematics(
        n_func,
        theory=theory,
        basis_type=basis_type,
        boundary_conditions=boundary_conditions,
        n_gauss=n_gauss,
    )

    print_simulation_setup(
        n_func,
        lambdas,
        delta_t,
        t_end,
        theory,
        boundary_conditions,
        basis_type,
        panel.n_gauss,
        parallel,
        num_proc,
    )

    # Parameters for nondimensional results
    h = panel.laminate.total_thickness
    E1 = AS4_PEEK.E if hasattr(AS4_PEEK, "E") else AS4_PEEK.nu21
    nu12 = AS4_PEEK.nu if hasattr(AS4_PEEK, "nu") else AS4_PEEK.nu12
    nu21 = AS4_PEEK.nu if hasattr(AS4_PEEK, "nu") else AS4_PEEK.nu21
    D011 = E1 * h**3 / 12 / (1 - nu12 * nu21)
    w0 = (D011 / (AS4_PEEK.rho * h * panel.a**4)) ** 0.5

    analyses = Analysis(panel)
    logger.info("Flutter analysis: ")
    start = timer()
    analyses.run_lambda_sweep(lambda_min=100, lambda_max=200, n_points=200)
    end = timer()
    timedif = (end - start) / 60  # minutes

    time_h = int(timedif // 60)

    remain = timedif % 60

    time_m = int(remain // 1)

    time_s = int(round((remain % 1) * 60))

    logger.info(
        f"Finished. Total Elapsed Time: "
        f"{time_h:d} h {time_m:d} m {time_s:d} s",
    )
    analyses.identify_flutter()
    print_terminal_summary_critical(analyses.lamb_cr_interp)

    panel_template = copy.deepcopy(panel)

    start = timer()

    if parallel:
        worker = partial(
            _single_simulation_worker,
            panel_template=panel_template,
            t_end=t_end,
            delta_t=delta_t,
        )

        # Security check for processors
        max_proc = mp.cpu_count() - 1
        requested_proc = num_proc
        n_proc = (
            min(max_proc, requested_proc) if requested_proc > 0 else max_proc
        )

        logger.info(
            f"Parallel batch with {len(lambdas)} cases, "
            f"using {n_proc} processors:"
        )

        with mp.Pool(processes=n_proc) as pool:
            results = pool.map(worker, lambdas)
    else:
        logger.info(f"Serial batch with {len(lambdas)} cases: ")
        results = [
            _single_simulation_worker(
                lamb,
                panel_template,
                delta_t,
                t_end,
            )
            for lamb in lambdas
        ]
    end = timer()
    timedif = (end - start) / 60  # minutes

    time_h = int(timedif // 60)

    remain = timedif % 60

    time_m = int(remain // 1)

    time_s = int(round((remain % 1) * 60))

    logger.info(
        f"Finished. Total Elapsed Time: "
        f"{time_h:d} h {time_m:d} m {time_s:d} s",
    )

    _, local_amp, _, _, _, _ = map(np.array, zip(*results))

    logger.info("\n" + "Post-critical analysis: ")
    print_terminal_summary_amplitude(local_amp)

    lambdas_sim = np.insert(lambdas, 0, analyses.lamb_cr_interp)
    local_amp_sim = np.insert(local_amp, 0, 0)

    plt.figure(1, figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
    plt.plot(
        list(TSUNEMATSU_DATABASE.keys()),
        list(TSUNEMATSU_DATABASE.values()),
        "--^b",
        markersize=3.5,
        label=r"Tsunematsu et al. (2021)",
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
    if save_figures:
        plt.savefig(
            str(output_dir / "limit_cycle_DixonMei.pdf"), bbox_inches="tight"
        )

    for kr, result in enumerate(results):
        h = panel.laminate.total_thickness
        _, _, time, qt, dqt_dt, fnl = result
        ws = (
            analyses.panel.get_displacement_at_points(
                np.array([[0.5, 0.0]]), 2, qt
            )
            / h
        )

        wps = analyses.panel.get_displacement_at_points(
            np.array([[0.5, 0.0]]), 2, dqt_dt
        ) / (h * w0)

        plt.figure(kr * 3 + 2, figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
        plt.plot(time, ws, "b")
        plt.gca().grid(
            visible=True, which="both", linestyle=":", linewidth=0.5
        )
        plt.ylabel("$w$ $[-]$", fontsize=10)
        plt.xlabel("Time $[s]$", fontsize=10)
        plt.title(rf"$@ (\xi,\eta)=(0.5,0)$: $\lambda={lambdas[kr]:.2f}$")
        if save_figures:
            plt.savefig(
                str(output_dir / f"w_lambda_{lambdas[kr]:d}.pdf"),
                bbox_inches="tight",
            )

        U = panel.evaluate_strain_energy(qt, fnl=-fnl)
        Ubar = U / D011 / ((h / panel.a) ** 2)
        timebar = time * w0

        T = panel.evaluate_kinetic_energy(dqt_dt)
        Tbar = T / D011 / ((h / panel.a) ** 2)

        Lbar = Tbar - Ubar
        Ebar = Ubar + Tbar

        fig = plt.figure(
            kr * 3 + 3, figsize=(FIGWIDTH, 2 * FIGHEIGHT), dpi=300
        )
        ax = fig.add_subplot(4, 1, 1)
        ax.plot(timebar, Ubar, "-k", linewidth=0.5)
        ax.grid(visible=True, which="both", linestyle=":", linewidth=0.5)
        ax.set_ylabel(r"$\overline{U}$ $[-]$", fontsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)

        ax = fig.add_subplot(4, 1, 2)
        ax.plot(timebar, Tbar, "-k", linewidth=0.5)
        ax.grid(visible=True, which="both", linestyle=":", linewidth=0.5)
        ax.set_ylabel(r"$\overline{T}$ $[-]$", fontsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)

        ax = fig.add_subplot(4, 1, 3)
        ax.plot(timebar, Ebar, "-k", linewidth=0.5)
        ax.grid(visible=True, which="both", linestyle=":", linewidth=0.5)
        ax.set_ylabel(r"$\overline{E}$ $[-]$", fontsize=10)
        plt.setp(ax.get_xticklabels(), visible=False)

        ax = fig.add_subplot(4, 1, 4)
        ax.plot(timebar, Lbar, "-k", linewidth=0.5)
        ax.grid(visible=True, which="both", linestyle=":", linewidth=0.5)
        ax.set_ylabel(r"$\overline{L}$ $[-]$", fontsize=10)
        ax.set_xlabel(r"$\tau$ $[-]$", fontsize=10)

        fig.align_ylabels()

        if save_figures:
            plt.savefig(
                str(output_dir / f"energy_{lambdas[kr]:d}.pdf"),
                bbox_inches="tight",
            )

        # Plotting only LCO data
        last_cycles = 5
        zero_crossings = np.where(np.diff(np.sign(ws)))[0]
        needed_crossings = 2 * last_cycles
        n_steps = len(ws)

        if len(zero_crossings) > needed_crossings:
            t_start = zero_crossings[-(needed_crossings + 1)]
            t_end = zero_crossings[-1]
        else:
            # Fallback to the last 20% of simulation.
            t_start = int(0.8 * n_steps)
            t_end = n_steps

        plt.figure(kr * 3 + 4, figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
        plt.plot(ws, wps, "-k", linewidth=0.5, label="Entire Simulation")
        plt.plot(ws[t_start:t_end], wps[t_start:t_end], "r", label="LCO cycle")
        plt.gca().grid(
            visible=True, which="both", linestyle=":", linewidth=0.5
        )
        plt.ylabel(r"$\dot{w}$ $[\tau^{-1}]$", fontsize=10)
        plt.xlabel(r"$w$ $[-]$", fontsize=10)
        plt.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=2,
            facecolor="white",
            framealpha=1,
            fontsize=10,
        )
        plt.title(
            rf"$@ (\xi,\eta)=(0.5,0)$: $\lambda={lambdas[kr]:.2f}$", y=1.15
        )

        if save_figures:
            plt.savefig(
                str(output_dir / f"phase_diagram_{lambdas[kr]:d}.pdf"),
                bbox_inches="tight",
            )

        if save_animation:
            analyses.create_3D_animation(
                qt,
                np.max(ws) * 1.1,
                pth2save=output_dir,
                file_name=f"animation_{lambdas[kr]:d}",
                str_title=rf"$\lambda={lambdas[kr]:.2f}$",
                show_arrow=show_arrow,
            )

    if not save_figures:
        plt.show()


if __name__ == "__main__":
    mp.freeze_support()
    # Execute benchmark case
    run_simulations(
        n_func=6,
        basis_type=BasisFunction.BARDELL,
        theory=StructuralTheory.REISSNER_MINDLIN,
        n_gauss=-1,
        lambdas=[
            key for key, value in DIXON_MEI_DATABASE.items() if value > 0.0
        ],
        delta_t=1e-6,
        t_end=0.1,
        parallel=False,
        save_figures=True,
        save_animation=True,
        show_arrow=True,
    )
