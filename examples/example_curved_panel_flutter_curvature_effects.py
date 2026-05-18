# %%
"""AE-250: Aeroelasticity Comprehensive Examples - Slide 02.

This script contains a suite of examples for validating panel flutter,
and vibration modes of curved aerospace panels.
All examples follow the AE-250 Course curriculum (ITA).
"""

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


if __name__ == "__main__":
    print("--- Starting AE-250 Examples ---\n")

    # Execute specific examples:
    # Example 1:
    run_flutter_analysis(radius_ratio=1.0)
    # Example 2:
    run_flutter_analysis(radius_ratio=1.25)

    input("Press Enter to exit...")

# %%
