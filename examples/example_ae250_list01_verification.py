"""Exercices List of class AE-250.

This scripts solves first exercices List of class AE-250.
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


def question_01(nfunc: int):
    """Function to run the first question of the list.

    Args:
        nfunc (int): Number of basis functions in each direction
    """
    #                     rho,    E,  nu
    AL = Isotropic("AL", 2700, 70e9, 0.3, alpha=21e-6, damping=0.05)

    laminate = Laminate("Aluminio")
    laminate.add_stack(AL, 1e-3)

    #                a,   b
    panel = Panel(0.3, 0.25, laminate=laminate)
    panel.setup_kinematics(
        nfunc,
        theory=StructuralTheory.KIRCHHOFF,
        basis_type=BasisFunction.SINES,
        n_gauss=10,
    )

    # Adding torsional edges
    panel.add_torsional_edge("left", 1e2)
    panel.add_torsional_edge("right", 1e2)

    # Adding linear spring at the center of the panel
    panel.add_linear_spring([0, 0], 5)

    panel.compute_free_modes()
    panel.plot_free_modes(n_modes=4)
    omega_L = panel.free_omega_hz[0] * 2 * np.pi
    omega_H = panel.free_omega_hz[3] * 2 * np.pi

    panel.compute_rayleigh_damping(omega_low=omega_L, omega_high=omega_H)

    # Initializing the analysis class with the panel
    analyses = Analysis(panel)

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
            + f" \tVelocity={np.round(analyses.v_inf_cr_interp, 2)} m/s,\t"
            + f"Mach={np.round(analyses.mach_cr_interp, 3)}"
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
    plt.gca().grid(visible=True, which="both", linestyle=":", linewidth=0.5)
    plt.xticks([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    plt.ylabel("Flutter Velocity $[m/s]$", fontsize=10)
    plt.xlabel(r"$\theta$ $[{}^\circ]$", fontsize=10)
    plt.show(block=False)

    N = np.array([-837, 0, 0])
    panel.compute_stiffness_given_pre_stress(N)
    analyses.run_flutter_sweep(n_points=500, n_modes_save=4)
    analyses.identify_flutter()
    analyses.plot_flutter_curves()

    print(
        f"Nx = {N[0]:.2f} N/m, Ny = {N[1]:.2f} N/m, Nxy = {N[2]:.2f} N/m, "
        + f"Flutter: \t{np.round(analyses.v_inf_cr_interp, 2)} m/s"
    )
    print("")

    t_critical = panel.run_thermal_buckling()

    print(f"Critical Buckling Temperature (DeltaTc): {t_critical:.2f} oC")


def question_02a(nfunc: int, Hs: list, thetas: list):
    """Function to run the second question of the list.

    Args:
        nfunc (int): Number of basis functions in each direction
        Hs (list): List of stiffener heights
        thetas (list): List of angles
    """
    #                     rho,    E,  nu
    AL = Isotropic("AL", 2700, 70e9, 0.3, alpha=21e-6, damping=0.05)
    # simulation

    laminate = Laminate("Aluminio")
    laminate.add_stack(AL, 5e-4)

    #                a,   b
    panel = Panel(0.3, 0.2, laminate, radius=1.0)
    panel.setup_kinematics(
        nfunc,
        theory=StructuralTheory.KIRCHHOFF,
        basis_type=BasisFunction.SINES,
        n_gauss=30,
    )

    base = Laminate("Base")
    base.add_stack(AL, 30e-3)

    flange = Laminate("Flange")
    flange.add_stack(AL, 1e-3)

    vfluter = np.zeros(shape=(len(Hs), len(thetas)))
    mfluter = np.zeros(shape=(len(Hs), len(thetas)))

    for kh, h in enumerate(Hs):
        h = h * panel.laminate.total_thickness
        for kt, theta in enumerate(thetas):
            panel.remove_stiffeners()

            panel.insert_stiffener(
                x0=0.0,
                y0=0.15,
                x1=0.3,
                y1=0.15,
                height=1e-3,
                laminate=base,
                gap=h,
                side=-1,
            )

            panel.insert_stiffener(
                x0=0.0,
                y0=0.15,
                x1=0.3,
                y1=0.15,
                height=20e-3,
                laminate=flange,
                gap=h + 1e-3,
                side=-1,
            )

            if h > 0.0:
                pad = Laminate("Pad")
                # Ensure increase is zero for theta 90 degrees
                d_h = 0.0 if theta == 90.0 else h / np.tan(np.radians(theta))
                pad.add_stack(AL, 30e-3 + d_h)
                panel.insert_stiffener(
                    x0=0.0,
                    y0=0.15,
                    x1=0.3,
                    y1=0.15,
                    height=h,
                    laminate=pad,
                    gap=0.0,
                    side=-1,
                )

            panel.compute_free_modes()
            omega_L = panel.free_omega_hz[0] * 2 * np.pi
            omega_H = panel.free_omega_hz[3] * 2 * np.pi
            panel.compute_rayleigh_damping(
                omega_low=omega_L, omega_high=omega_H
            )

            analyses = Analysis(panel)

            analyses.set_atmosphere(1e4)

            analyses.run_flutter_sweep(
                mach_max=10, n_points=500, n_modes_save=4
            )
            analyses.identify_flutter()

            vfluter[kh, kt] = analyses.v_inf_cr_interp
            mfluter[kh, kt] = analyses.mach_cr_interp

            print(
                f"H = {h * 1e3:.2f} mm, theta = {theta:.1f} o, flutter: "
                + f"Velocity = {vfluter[kh, kt]:.2f} m/s, "
                + f"Mach = {mfluter[kh, kt]:.2f}"
            )

    colors = ["--ob", "--pr", "--sk", "--dm", "--Py", "--^g"]

    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
    for kh, theta in enumerate(thetas):
        plt.plot(
            np.array(Hs) * panel.laminate.total_thickness,
            mfluter[kh, :],
            colors[kh],
            markersize=3.5,
            label=r"$\theta = %d~^{\circ}$" % theta,
        )
    plt.gca().grid(visible=True, which="both", linestyle=":", linewidth=0.5)
    plt.xticks(thetas)
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=3,
        facecolor="white",
        framealpha=1,
        handlelength=1.7,
        markerscale=1,
    )
    plt.ylabel("Flutter Mach $[-]$", fontsize=10)
    plt.xlabel(r"$H$ $[mm]$", fontsize=10)
    plt.show(block=False)

    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT), dpi=300)
    for kh, h in enumerate(Hs):
        plt.plot(
            thetas,
            vfluter[kh, :],
            colors[kh],
            markersize=3.5,
            label=r"$H = %.1f~h_{panel}$" % h,
        )
    plt.gca().grid(visible=True, which="both", linestyle=":", linewidth=0.5)
    plt.xticks(thetas)
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=2,
        facecolor="white",
        framealpha=1,
        handlelength=1.7,
        markerscale=1,
    )
    plt.ylabel("Flutter Velocity $[m/s]$", fontsize=10)
    plt.xlabel(r"$\theta$ $[{}^\circ]$", fontsize=10)
    plt.show(block=False)

    return panel


def question_02b(nfunc: int):
    """Function to run the second question of the list.

    Args:
        nfunc (int): Number of basis functions in each direction
        Hs (list): List of stiffener heights
        thetas (list): List of angles
    """
    #                     rho,    E,  nu
    AL = Isotropic("AL", 2700, 70e9, 0.3, alpha=21e-6, damping=0.05)
    # setup de simulacao

    laminate = Laminate("Aluminio")
    laminate.add_stack(AL, 5e-4)

    #                a,   b
    panel = Panel(0.3, 0.2, laminate, radius=1.0)
    panel.setup_kinematics(
        nfunc,
        theory=StructuralTheory.KIRCHHOFF,
        basis_type=BasisFunction.SINES,
        n_gauss=30,
    )

    base = Laminate("Base")
    base.add_stack(AL, 30e-3)

    flange = Laminate("Flange")
    flange.add_stack(AL, 1e-3)

    panel.insert_stiffener(
        x0=0.0,
        y0=0.15,
        x1=0.3,
        y1=0.15,
        height=1e-3,
        laminate=base,
        gap=0.0,
        side=-1,
    )

    panel.insert_stiffener(
        x0=0.0,
        y0=0.15,
        x1=0.3,
        y1=0.15,
        height=20e-3,
        laminate=flange,
        gap=1e-3,
        side=-1,
    )

    panel.compute_free_modes()
    panel.plot_free_modes(n_modes=4)
    omega_L = panel.free_omega_hz[0] * 2 * np.pi
    omega_H = panel.free_omega_hz[3] * 2 * np.pi
    panel.compute_rayleigh_damping(omega_low=omega_L, omega_high=omega_H)

    # 2. Natural Frequency variation with DeltaT
    delta_ts = np.linspace(20, 30, 50)
    n_modes_plot = 4
    omegas_history = np.zeros((n_modes_plot, len(delta_ts)))

    for i, dt in enumerate(delta_ts):
        panel.compute_thermal_stiffness(
            delta_t=dt, distribution=BasisFunction.SINES
        )
        panel.compute_free_modes()
        omegas_history[:, i] = panel.free_omega_hz[:n_modes_plot]

    # Plot Frequency vs DeltaT
    plt.figure(figsize=(6, 4), dpi=300)
    for k in range(n_modes_plot):
        plt.plot(delta_ts, omegas_history[k, :], label=f"Mode {k + 1}")
    plt.xlabel(r"$\Delta T$ [${}^\circ$C]", fontsize=12)
    plt.ylabel(r"$\omega$ [Hz]", fontsize=12)
    plt.title("Thermal Influence on Natural Frequencies")
    plt.legend(facecolor="white", framealpha=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    panel.run_thermal_buckling(delta_t0=800, distribution=BasisFunction.SINES)


if __name__ == "__main__":
    # Question 01 (a, b, c, d)
    # question_01(nfunc=4)

    # Question 02 (a, b)
    # Hs is the pad height in terms of fraction of panel total thickness
    Hs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    thetas = np.array([15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    # question_02a(nfunc=10, Hs=Hs, thetas=thetas)

    question_02b(nfunc=8)

    input("Press Enter to finish...")

# %%
