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


def get_default_laminate():
    """Create a standard Aluminum 2024-T3 laminate for examples."""
    al_2024_t3 = Isotropic(
        "AL2024T3", 2700, 75e9, 0.3, alpha=30e-6, damping=0.03
    )
    laminate = Laminate("Aluminum_Skin")
    laminate.add_stack(al_2024_t3, 1e-3)
    return laminate


def run_flutter_analysis(radius_ratio: float):
    """Perform flutter analysis for a given panel curvature.

    Args:
        radius_ratio (float): The curvature ratio (1/R).
    """
    n_func = 2
    laminate = get_default_laminate()

    # Panel setup: a=0.3, b=0.25
    panel = Panel(
        0.3,
        0.25,
        laminate,
        radius=1 / radius_ratio if radius_ratio != 0 else 0,
    )
    panel.setup_kinematics(
        n_func,
        theory=StructuralTheory.KIRCHHOFF,
        basis_type=BasisFunction.SINES,
        n_gauss=15,
    )

    sim = Analysis(panel)
    sim.set_atmosphere(altitude=1e4)  # 10km altitude

    # Run flutter sweep up to Mach 10
    sim.run_flutter_sweep(mach_max=10.0, n_points=500, n_modes_save=4)
    sim.identify_flutter()

    print(f"Curvature 1/R = {radius_ratio}")
    print(f"Critical Flutter Velocity: {sim.v_inf_cr_interp:.2f} m/s\n")

    sim.plot_flutter_curves()
    panel.compute_free_modes()
    panel.plot_free_modes(n_modes=4)


def run_thermal_example(radius_ratio: float):
    """Analyze thermal buckling and frequency degradation under Delta T.

    Args:
        radius_ratio (float): The curvature ratio (1/R).
    """
    n_func = 2
    laminate = get_default_laminate()
    panel = Panel(
        0.3,
        0.25,
        laminate,
        radius=1 / radius_ratio if radius_ratio != 0 else 0,
    )
    panel.setup_kinematics(
        n_func,
        theory=StructuralTheory.KIRCHHOFF,
        basis_type=BasisFunction.SINES,
        n_gauss=15,
    )

    sim = Analysis(panel)

    # 1. Critical Temperature Search
    # Compute critical thermal buckling temperature.
    dt_critical = panel.run_thermal_buckling()

    print(f"Critical Buckling Temperature: {dt_critical:.2f} °C\n")

    # 2. Frequency vs. Temperature Mapping
    delta_ts = np.linspace(-dt_critical, 1.5 * dt_critical, 100)
    n_modes_plot = 4
    omegas_history = np.zeros((n_modes_plot, len(delta_ts)))

    for i, dt in enumerate(delta_ts):
        panel.compute_thermal_stiffness(dt)
        panel.compute_free_modes()
        omegas_history[:, i] = panel.free_omega_hz[:n_modes_plot]

    # Plotting Results
    plt.figure(figsize=(8, 5))
    colors = ["b", "r", "k", "m"]
    for k in range(n_modes_plot):
        plt.plot(
            delta_ts, omegas_history[k, :], colors[k], label=f"Mode {k + 1}"
        )

    plt.axvline(
        dt_critical, color="gray", linestyle="--", label=r"$\Delta T_{cr}$"
    )
    plt.xlabel(r"$\Delta T$ [${}^\circ C$]")
    plt.ylabel(r"$\omega$ [Hz]")
    plt.legend(facecolor="white", framealpha=1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # 3. Flutter analysis at specific thermal states
    sim.set_atmosphere(altitude=10000)  # Analysis at 10km altitude

    thermal_steps = np.array([-0.20, 0.0, 0.20])
    for dt_step in thermal_steps:
        panel.compute_thermal_stiffness(dt_step * dt_critical)
        # Re-run flutter sweep for each thermal condition
        if radius_ratio == 1.25 and dt_step != 0.0:
            mach_max = 25.0
        else:
            mach_max = 6.0
        if radius_ratio == 0.1:
            mach_max = 5.0
        sim.run_flutter_sweep(mach_max=mach_max, n_points=500, n_modes_save=4)
        sim.identify_flutter()

        print(
            f"DeltaT = {dt_step * dt_critical:.2f} oC | Flutter Speed:"
            + f" {sim.v_inf_cr_interp:.2f} m/s"
        )
        sim.plot_flutter_curves(xaxis="V_inf")


if __name__ == "__main__":
    print("--- Starting AE-250 Examples ---\n")

    # Execute specific examples:
    # Example 1:
    # run_flutter_analysis(radius_ratio=1.0)
    # Example 2:
    # run_flutter_analysis(radius_ratio=1.25)

    # Example 3:
    # run_thermal_example(radius_ratio=0.1)
    # Example 4:
    run_thermal_example(radius_ratio=1.25)

# %%
