"""Exercices List of class AA-250.

This scripts solves first exercices List of class AA-250.
"""

# %%
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

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", weight="bold")
plt.rcParams["grid.alpha"] = 1
plt.rcParams["figure.facecolor"] = "1"
plt.rcParams["legend.fontsize"] = "12"

CM_TO_INCH = 0.393701
WIDTH = 13.0
FIGWIDTH = WIDTH * CM_TO_INCH
FIG_RATIO = 0.65
FIGHEIGHT = FIGWIDTH * FIG_RATIO

if __name__ == "__main__":
    #                     rho,    E,  nu
    AL = Isotropic("AL", 2700, 70e9, 0.3, alpha=21e-6, damping=0.05)

    question = 2

    if question == 1:
        """
        EXERCICIO 01
        """

        # setup de simulacao
        n_func = 4

        laminado = Laminate("Aluminio")
        laminado.add_stack(AL, 1e-3)

        #                a,   b
        painel = Panel(0.3, 0.25, laminado)
        painel.setup_kinematics(
            n_func,
            theory=StructuralTheory.KIRCHHOFF,
            basis_type=BasisFunction.SINES,
            n_gauss=30,
        )

        # Adicionando as molas torcionais
        painel.add_torsional_edge("left", 1e2)
        painel.add_torsional_edge("right", 1e2)

        # Adicionando mola linear
        painel.add_linear_spring([0, 0], 5)

        painel.compute_free_modes()
        painel.plot_free_modes(n_modes=4)
        omega_L = painel.free_omega_hz[0] * 2 * np.pi
        omega_H = painel.free_omega_hz[3] * 2 * np.pi

        painel.compute_rayleigh_damping(omega_low=omega_L, omega_high=omega_H)

        # Iniciando as analises
        analyses = Analysis(painel)

        analyses.set_atmosphere(1e4)

        thetas = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

        vfluter = np.zeros(shape=(len(thetas), 1))

        for kt, theta in enumerate(thetas):
            analyses.run_flutter_sweep(
                n_points=500, theta_flow=theta, n_modes_save=4
            )
            analyses.identify_flutter()
            analyses.plot_flutter_curves()

            print(
                f"theta = {np.round(theta, 2)}, Flutter:"
                + f" \t{np.round(analyses.v_inf_cr_interp, 2)} m/s"
            )
            print("")

            vfluter[kt] = analyses.v_inf_cr_interp

        plt.figure(figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
        plt.plot(
            thetas,
            vfluter,
            "--ok",
            markersize=3.5,
        )
        plt.gca().grid(
            visible=True, which="both", linestyle=":", linewidth=0.5
        )
        plt.xticks([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        plt.ylabel("Flutter Velocity $[m/s]$", fontsize=10)
        plt.xlabel(r"$\theta$ $[{}^\circ]$", fontsize=10)
        plt.show(block=False)

        t_critical = painel.run_thermal_buckling()

    if question == 2:
        """
        EXERCICIO 02
        """
        # setup de simulacao
        n_func = 2

        laminado = Laminate("Aluminio")
        laminado.add_stack(AL, 5e-4)

        #                a,   b
        painel = Panel(0.3, 0.2, laminado, radius=1.0)
        painel.setup_kinematics(
            n_func,
            theory=StructuralTheory.KIRCHHOFF,
            basis_type=BasisFunction.SINES,
            n_gauss=30,
        )

        painel.compute_free_modes()
        omega_L = painel.free_omega_hz[0] * 2 * np.pi
        omega_H = painel.free_omega_hz[3] * 2 * np.pi

        painel.compute_rayleigh_damping(omega_low=omega_L, omega_high=omega_H)

        """
        base = Laminate("Base")
        base.add_stack(AL, 30e-3)

        flange = Laminate("Flange")
        flange.add_stack(AL, 1e-3)

        painel.insert_stiffener(
            x0=0.0,
            y0=0.15,
            x1=0.3,
            y1=0.15,
            height=1e-3,
            laminate=base,
            gap=0.0,
            side=-1,
        )

        painel.insert_stiffener(
            x0=0.0,
            y0=0.15,
            x1=0.3,
            y1=0.15,
            height=20e-3,
            laminate=flange,
            gap=1e-3,
            side=-1,
        )
        """

        painel.compute_free_modes()
        painel.plot_free_modes(n_modes=4)

        """
        t_critical = painel.run_thermal_buckling(
            distribution=BasisFunction.SINES
        )

        painel.compute_thermal_stiffness(
            delta_t=0.4, distribution=BasisFunction.SINES
        )
        """

        analyses = Analysis(painel)

        analyses.set_atmosphere(1e4)

        analyses.run_flutter_sweep(mach_max=10, n_points=500, n_modes_save=4)
        analyses.identify_flutter()
        analyses.plot_flutter_curves()


# %%
