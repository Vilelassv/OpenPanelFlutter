# %%
"""AE-250: Aeroelasticity Comprehensive Examples - Slide 01.

This script solves pedagogical examples for panel flutter.
Example 01: Flutter analysis using Ackeret's Model vs. with damping.
All examples follow the AE-250 Course curriculum (ITA).
"""

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
    """Run first example for AE-250 slide 01 validation."""
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
    EXAMPLE 01: Flutter analysis at Sea Level (ISA 0m).
    Compares Ackeret's Theory vs. Damped Model.
    """
    # Calculate frequencies for Rayleigh Damping setup
    panel.compute_free_modes()
    omega_low = panel.free_omega_hz[0] * 2 * np.pi
    omega_high = panel.free_omega_hz[3] * 2 * np.pi
    panel.compute_rayleigh_damping(omega_low=omega_low, omega_high=omega_high)

    sim = Analysis(panel)
    sim.set_atmosphere(altitude=0)

    # Scenario A: Ackeret (Un-damped)
    sim.run_flutter_sweep(use_ackeret=True, n_points=500, n_modes_save=4)
    sim.identify_flutter()
    print(f"Ackeret Flutter Velocity: {sim.v_inf_cr_interp:.2f} m/s")
    sim.plot_flutter_curves()

    # Scenario B: (Damped)
    sim.run_flutter_sweep(use_ackeret=False, n_points=500, n_modes_save=4)
    sim.identify_flutter()
    print(f"Damped Flutter Velocity:  {sim.v_inf_cr_interp:.2f} m/s")
    sim.plot_flutter_curves()

    panel.plot_free_modes(n_modes=4)


if __name__ == "__main__":
    run_examples()
    input("Press Enter to exit...")

# %%
