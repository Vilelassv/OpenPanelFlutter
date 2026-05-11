# %%
"""AE-250: Aeroelasticity Comprehensive Examples - Slide 01.

This script solves pedagogical examples for panel flutter and thermal buckling.
Example 02: Effect of thermal loads on natural frequencies and flutter onset.
All examples follow the AE-250 Course curriculum (ITA).
"""

import matplotlib.pyplot as plt
import numpy as np

from openpanelflutter.analysis import Analysis
from openpanelflutter.definitions import (
    BasisFunction,
    StructuralTheory,
    apply_plot_style,
)
from openpanelflutter.material import Isotropic, Laminate
from openpanelflutter.panel import Panel

# Configuring the global Matplotlib style
apply_plot_style()


def run_examples():
    """Run second example for AE-250 slide 01 validation."""
    # Simulation Setup
    n_func = 2
    fnc_disp = BasisFunction.SINES
    theory = StructuralTheory.KIRCHHOFF
    n_gauss = 15

    # 1. Material and Laminate Setup
    # Aluminum 2024-T3: rho, E, nu, alpha_thermal, damping_zeta
    al_2024_t3 = Isotropic(
        "AL2024T3", 2700, 75e9, 0.3, alpha=30e-6, damping=0.03
    )

    laminate = Laminate("Aluminum_Skin")
    laminate.add_stack(al_2024_t3, 1e-3)

    # 2. Panel Geometry
    # Square panel: a=0.2m, b=0.2m, laminate
    panel = Panel(0.2, 0.2, laminate)
    panel.setup_kinematics(
        n_func, theory=theory, basis_type=fnc_disp, n_gauss=n_gauss
    )

    """
    EXAMPLE 02 (class 01): Thermal-Aeroelastic interaction.
    Analyzes the influence of temperature on frequencies and flutter.
    """
    sim = Analysis(panel)

    # 1. Thermal Buckling Analysis
    dt_critical = panel.run_thermal_buckling()

    print(f"Critical Buckling Temperature (DeltaTc): {dt_critical:.2f} oC")

    # 2. Natural Frequency variation with DeltaT
    delta_ts = np.linspace(-dt_critical, 1.5 * dt_critical, 100)
    n_modes_plot = 4
    omegas_history = np.zeros((n_modes_plot, len(delta_ts)))

    for i, dt in enumerate(delta_ts):
        panel.compute_thermal_stiffness(dt)
        panel.compute_free_modes()
        omegas_history[:, i] = panel.free_omega_hz[:n_modes_plot]

    # Plot Frequency vs DeltaT
    plt.figure(figsize=(6, 4), dpi=300)
    for k in range(n_modes_plot):
        plt.plot(delta_ts, omegas_history[k, :], label=f"Mode {k + 1}")

    plt.axvline(
        dt_critical, color="r", linestyle="--", label=r"$\Delta T_{cr}$"
    )
    plt.xlabel(r"$\Delta T$ [${}^\circ$C]", fontsize=12)
    plt.ylabel(r"$\omega$ [Hz]", fontsize=12)
    plt.title("Thermal Influence on Natural Frequencies")
    plt.legend(facecolor="white", framealpha=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # 3. Flutter analysis at specific thermal states
    sim.set_atmosphere(altitude=10000)  # Analysis at 10km altitude

    thermal_steps = np.array([0.0, 0.5, 0.98]) * dt_critical
    for dt_step in thermal_steps:
        panel.compute_thermal_stiffness(dt_step)
        # Re-run flutter sweep for each thermal condition
        sim.run_flutter_sweep(mach_max=10.0, n_points=500, n_modes_save=4)
        sim.identify_flutter()

        print(
            f"DeltaT = {dt_step:.2f} oC | Flutter Speed:"
            + f" {sim.v_inf_cr_interp:.2f} m/s"
        )
        sim.plot_flutter_curves(xaxis="V_inf")


if __name__ == "__main__":
    run_examples()
    input("Press Enter to exit...")

# %%
