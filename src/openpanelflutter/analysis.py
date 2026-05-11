"""Analysis module for aeroelastic stability and atmospheric modeling.

This module provides the Analysis class to perform flutter sweeps,
divergence analysis, and atmospheric property calculations for
aerospace structures.
"""

import multiprocessing as mp
from functools import partial
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from .definitions import apply_plot_style
from .integrator import Integrator

apply_plot_style()


class Analysis:
    """Handles aeroelastic simulations and atmospheric data.

    Attributes:
        panel: A Panel object containing structural and aerodynamic matrices.
        h (float): Flight altitude [m].
        rho_air (float): Air density [kg/m^3].
        v_sound (float): Speed of sound [m/s].
        dt_thermal (float): Temperature difference for thermal stress [K].
    """

    def __init__(self, panel):
        """Initialize the analysis with a reference to a Panel object."""
        self.panel = panel

        # Atmospheric properties
        self.h = 0.0
        self.rho_air = 1.225
        self.v_sound = 340.0
        self.dt_thermal = 0.0

        # Results storage
        self.machs = None
        self.v_infs = None
        self.lambdas = None
        self.omega_hz = None
        self.damping_zeta = None
        self.img_omega = None
        self.modes_history = None

    def set_atmosphere(self, altitude: float, t_ref: float = 25.0):
        """Compute atmospheric properties based on altitude.

        Args:
            altitude (float): Altitude above sea level [m].
            t_ref (float): Reference temperature for thermal stress [oC].
        """
        self.h = altitude
        # International Standard Atmosphere (ISA) model
        temp_k = 288.15 - 0.0065 * self.h
        t_ref_k = t_ref + 273.15

        self.rho_air = 1.225 * (temp_k / 288.15) ** 4.2558774
        self.v_sound = np.sqrt(1.4 * 287.0 * temp_k)

        # Temperature delta for thermal stiffness calculations
        self.dt_thermal = temp_k - t_ref_k

    def run_flutter_sweep(
        self,
        mach_min: float = np.sqrt(2),
        mach_max: float = 5.0,
        n_points: int = 50,
        n_modes_save: int = 4,
        theta_flow: float = 0.0,
        use_ackeret: bool = False,
    ):
        """Perform a flutter sweep over a range of Mach numbers.

        Args:
            mach_min (float): Starting Mach number.
            mach_max (float): Ending Mach number.
            n_points (int): Number of points in the sweep.
            n_modes_save (int): Number of modes to store in history.
            theta_flow (float): Flow angle [degree].
            use_ackeret (bool): If True, ignores damping.
        """
        mach_array = np.linspace(mach_min, mach_max, n_points)
        v_infs = mach_array * self.v_sound

        # Convert flow angle to radians for calculations
        theta_flow = np.radians(theta_flow)

        # Pre-calculating aerodynamic stiffness based on flow angle
        # Combining Kaer (x-direction) and Kaery (y-direction)
        kaer_combined = self.panel._K_aer * np.cos(
            theta_flow
        ) + self.panel._K_aery * np.sin(theta_flow)

        num_dofs = self.panel.len_tot
        self.omega_hz = np.zeros((n_points, num_dofs))
        self.damping_zeta = np.zeros((n_points, num_dofs))
        self.img_omega = np.zeros((n_points, num_dofs))
        self.modes_history = np.zeros(
            (n_points, num_dofs, n_modes_save), dtype=np.complex128
        )

        # Constant for Ackeret theory (no damping)
        damp_toggle = 0.0 if use_ackeret else 1.0
        self.use_ackeret = use_ackeret

        # Structural matrices (already summed from Panel @properties)
        m_mat = self.panel.M_global
        m_inv = la.inv(m_mat)
        k_str = self.panel.K_structural
        c_str = self.panel.C_structural

        curvature = 1.0 / self.panel.radius if self.panel.radius > 0.0 else 0.0

        for i, mach in enumerate(mach_array):
            # Piston Theory Parameters
            beta = np.sqrt(mach**2 - 1)
            q_dyn = 0.5 * self.rho_air * v_infs[i] ** 2

            # Aerodynamic pressure (lambda) and damping (chi)
            lam = 2.0 * q_dyn / beta
            chi = (2.0 * q_dyn / (v_infs[i] * beta)) * (
                (mach**2 - 2) / (mach**2 - 1)
            )
            gamma = lam / 2 * curvature / beta

            # Total Matrices for current state
            k_tot = (
                k_str + (lam * kaer_combined) + (-gamma * self.panel._K_aer_r)
            )
            c_tot = damp_toggle * (c_str + (chi * self.panel._C_caer))

            # State-space: [I 0; 0 M] * {dot_x} = [0 I; -K -C] * {x}
            # Simplified here to: {dot_z} = [0 I; -M^-1*K -M^-1*C] * {z}

            state_matrix = np.zeros((2 * num_dofs, 2 * num_dofs))
            state_matrix[:num_dofs, num_dofs:] = np.eye(num_dofs)
            state_matrix[num_dofs:, :num_dofs] = -m_inv @ k_tot
            state_matrix[num_dofs:, num_dofs:] = -m_inv @ c_tot

            # Solve complex eigenvalue problem
            eigvals, eigvecs = la.eig(state_matrix)

            # Extract upper half (displacements) and one of each conjugate pair
            eigvals = eigvals[::2]
            mode_shapes = eigvecs[:num_dofs, ::2]

            # Sort by frequency magnitude
            sort_idx = np.abs(eigvals).argsort()
            eigvals = eigvals[sort_idx]
            mode_shapes = mode_shapes[:, sort_idx]

            # Store results
            self.omega_hz[i, :] = np.abs(eigvals) / (2.0 * np.pi)
            self.damping_zeta[i, :] = np.real(eigvals) / np.abs(eigvals)
            self.img_omega[i, :] = np.imag(eigvals**2)
            self.modes_history[i, :, :] = mode_shapes[:, :n_modes_save]

        self.machs = mach_array
        self.v_infs = v_infs
        self.lambdas = (
            2.0 * (0.5 * self.rho_air * v_infs**2) / np.sqrt(mach_array**2 - 1)
        )

    def identify_flutter(self) -> float:
        """Identify flutter onset using a bracket-and-interpolate method.

        This method scans the damping across all analyzed modes to detect
        zero-crossings. When a crossing is found, it performs a local linear
        interpolation between the last stable and first unstable points to
        accurately determine the critical flutter parameters.

        Returns:
            float: The critical dynamic pressure (lambda_cr).
        """
        if self.damping_zeta is None:
            raise ValueError(
                "No analysis data available. Run the sweep first."
            )

        # Initialize with null values indicating "not found"
        self._set_critical_values(None, 0.0, 0.0, 0.0)

        tol = np.finfo(np.float32).eps
        lamb_cr_candidate = float("inf")
        found_flutter = False

        # Iterate through each saved mode to find the first instability onset
        for mode in range(self.modes_history.shape[2]):
            zeta_mode = self.damping_zeta[:, mode]

            # Find indices where a sign change occurs (from stable to unstable)
            # Logic: zeta[i-1] <= tol and zeta[i] > tol
            crossings = np.where(
                (zeta_mode[:-1] <= tol) & (zeta_mode[1:] > tol)
            )[0]

            if crossings.size > 0:
                first_cross_idx = crossings[0]

                # Index 'i': first unstable point, 'i-1': last stable point
                idx_stable = first_cross_idx
                idx_unstable = first_cross_idx + 1
                current_lamb_at_cross = self.lambdas[idx_unstable]

                # Check if this is the earliest flutter found so far
                if current_lamb_at_cross < lamb_cr_candidate:
                    found_flutter = True
                    lamb_cr_candidate = current_lamb_at_cross

                    # 1. Store discrete values (at the exact sweep point)
                    self.id_cr = idx_unstable
                    self.mach_cr = self.machs[idx_unstable]
                    self.v_inf_cr = self.v_infs[idx_unstable]
                    self.lamb_cr = current_lamb_at_cross

                    # 2. Linear Interpolation: only the two bracketing points
                    # The goal is to find the exact root where zeta(Mach) = 0
                    z_pair = self.damping_zeta[
                        [idx_stable, idx_unstable], mode
                    ]

                    self.mach_cr_interp = np.interp(
                        0.0, z_pair, self.machs[[idx_stable, idx_unstable]]
                    )
                    self.v_inf_cr_interp = np.interp(
                        0.0, z_pair, self.v_infs[[idx_stable, idx_unstable]]
                    )
                    self.lamb_cr_interp = np.interp(
                        0.0, z_pair, self.lambdas[[idx_stable, idx_unstable]]
                    )

        return self.lamb_cr if found_flutter else 0.0

    def _set_critical_values(self, id_val, mach, v_inf, lamb):
        """Helper to reset or set critical analysis values."""
        self.id_cr = id_val
        self.mach_cr = mach
        self.v_inf_cr = v_inf
        self.lamb_cr = lamb
        self.mach_cr_interp = mach
        self.v_inf_cr_interp = v_inf
        self.lamb_cr_interp = lamb

    def plot_flutter_curves(self, **kwargs):
        """Plot V-g and V-f diagrams with support for Ackeret theory.

        **kwargs: Additional parameters like:
                Vf (bool): If True, plots both V-g and V-f diagrams.
                    f False, plots only frequency. Default is True.
                xaxis (str): The variable for the x-axis ('V_inf', 'Mach',
                    or 'lambda'). Default is 'V_inf'.
        """
        if self.omega_hz is None:
            raise ValueError(
                "No analysis data available. Run the sweep first."
            )

        # Configuration and sizing
        cm_to_inch = 0.393701
        fig_width = 30.0 * cm_to_inch
        fig_height = fig_width * 0.65

        plt.figure(figsize=(fig_width, fig_height), dpi=300)

        show_dual_plot = kwargs.get("Vf", True)
        x_mode = kwargs.get("xaxis", "V_inf")

        # Define x-axis data and label
        if x_mode == "V_inf":
            x_data = self.v_infs
            x_label = r"$V_\infty$ [m/s]"
        elif x_mode == "Mach":
            x_data = self.machs
            x_label = r"Mach $[-]$"
        else:  # lambda
            x_data = self.lambdas
            x_label = r"$\lambda$ $[-]$"

        colors = [
            "b",
            "r",
            "k",
            "m",
            "y",
            "g",
            "--b",
            "--r",
            "--k",
            "--m",
            "--y",
            "--g",
        ]

        for k in range(self.modes_history.shape[2]):
            # 1. Frequency Plot (Real part/Magnitude)
            if show_dual_plot:
                plt.subplot(1, 2, 1)

            plt.plot(x_data, self.omega_hz[:, k], colors[k % len(colors)])
            plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(r"$\omega$ [Hz]", fontsize=12)

            # 2. Stability Plot (Damping or Imaginary Part)
            if show_dual_plot:
                plt.subplot(1, 2, 2)

                # If using Ackeret, plot the Imaginary part of frequency
                if hasattr(self, "use_ackeret") and self.use_ackeret:
                    y_data = self.img_omega[:, k]
                    y_label = r"Im$(\omega^2)$"
                else:
                    # Standard damping ratio (zeta)
                    y_data = self.damping_zeta[:, k]
                    y_label = r"$\zeta$ $[-]$"

                plt.plot(x_data, y_data, colors[k % len(colors)])
                plt.grid(
                    visible=True, which="both", linestyle="--", linewidth=0.5
                )
                plt.xlabel(x_label, fontsize=12)
                plt.ylabel(y_label, fontsize=12)

                # Add a reference line for stability threshold
                if not self.use_ackeret:
                    plt.axhline(0, color="black", linewidth=0.8, alpha=0.7)

        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def _single_sim_worker(
        lamb_val, obj, tf, dt, Nmodes, theta_flow, kind, solver_type, kwargs
    ):
        """Internal worker function for parallel processing."""
        # This logic replicates the matrix assembly for a single lambda
        m_mat = obj.panel.M_global
        k_str = obj.panel.K_structural
        c_str = obj.panel.C_structural

        material = obj.panel.laminate.plies[0].material
        h = obj.panel.laminate.total_thickness
        E1 = material.E if hasattr(material, "E") else material.E1
        nu12 = material.nu if hasattr(material, "nu") else material.nu12
        nu21 = material.nu if hasattr(material, "nu") else material.nu21

        D011 = E1 * h**3 / 12 / (1 - nu12 * nu21)
        w0 = (D011 / (material.rho * h * obj.panel.a**4)) ** 0.5
        chi = ((lamb_val * 0.01) ** 0.5) * D011 / (w0 * obj.panel.a**4)

        kaer_combined = obj.panel._K_aer * np.cos(
            theta_flow
        ) + obj.panel._K_aery * np.sin(theta_flow)

        k_tot = k_str + (lamb_val * D011 / (obj.panel.a**3)) * kaer_combined
        c_tot = c_str + chi * obj.panel._C_caer

        integ = Integrator(
            obj.panel,
            tf,
            dt,
            m_mat,
            k_tot,
            c_tot,
            n_modes=Nmodes,
            kind=kind,
            **kwargs,
        )

        if solver_type.lower() == "gen_alpha":
            integ.solve_gen_alpha()
        else:
            integ.solve_cdm()
        return integ

    def post_flutter_sim(
        self, lamb, tf, dt, Nmodes, theta_flow=0.0, kind="NM", **kwargs
    ):
        """Run post-flutter simulations with optional parallelization.

        Args:
            lamb (float or list): Dynamic pressure parameter(s).
            tf (float): Final time.
            dt (float): Time step.
            Nmodes (int): Number of modes for reduction.
            theta_flow (float): Flow angle in radians.
            kind (str): Reduction type ('NM' or 'AEM').
            **kwargs:
                solver (str): 'CDM' (default) or 'gen_alpha'.
                parallel (bool): If True, runs multiple lambdas in parallel.
                num_proc (int): Number of processors to use.
        """
        solver_type = kwargs.get("solver", "CDM")
        is_parallel = kwargs.get("parallel", False)
        if not hasattr(self.panel, "NLNL"):
            self.panel.init_nlin()

        # Handle lambda as a list for parallel or single for loop
        lamb_list = [lamb] if np.isscalar(lamb) else lamb

        if is_parallel and len(lamb_list) > 1:
            # Security check for processors
            max_proc = mp.cpu_count() - 1
            requested_proc = kwargs.get("num_proc", max_proc)
            n_proc = (
                min(max_proc, requested_proc)
                if requested_proc > 0
                else max_proc
            )

            start = timer()
            print("Simulating...")
            mp.freeze_support()

            # Partial function to fix constant arguments
            worker = partial(
                self._single_sim_worker,
                obj=self,
                tf=tf,
                dt=dt,
                Nmodes=Nmodes,
                theta_flow=theta_flow,
                kind=kind,
                solver_type=solver_type,
                kwargs=kwargs,
            )

            with mp.Pool(processes=n_proc) as pool:
                results = pool.map(worker, lamb_list)
            end = timer()
            timedif = (end - start) / 60  # minutes

            time_h = timedif // 60

            remain = timedif % 60

            time_m = remain // 1

            time_s = round((remain % 1) * 60)

            print(
                "Finished, Total Elapsed Time: ",
                "%d h %d m % d s" % (time_h, time_m, time_s),
            )
        else:
            # Sequential execution
            start = timer()
            print("Simulating...")
            results = [
                self._single_sim_worker(
                    lamb,
                    self,
                    tf,
                    dt,
                    Nmodes,
                    theta_flow,
                    kind,
                    solver_type,
                    kwargs,
                )
                for lamb in lamb_list
            ]
            end = timer()
            timedif = (end - start) / 60  # minutes

            time_h = timedif // 60

            remain = timedif % 60

            time_m = remain // 1

            time_s = round((remain % 1) * 60)

            print(
                "Finished, Total Elapsed Time: ",
                "%d h %d m % d s" % (time_h, time_m, time_s),
            )
        self.results = results
        wA = np.zeros(shape=(len(results), 1))
        wA_p = np.zeros(shape=(len(results), 1))
        for kr, result in enumerate(results):
            wA[kr], _, _ = self.max_lco_amplitude(result.panel, result.displ)
            wA_p[kr] = self.get_point_amplitude(
                result.panel, result.displ, np.array([0.5, 0.0])
            )
        return wA, wA_p

    def max_lco_amplitude(
        self, panel, disp_history, last_cycles=2, chunk_size=100
    ):
        """Calculate the maximum LCO amplitude and its location.

        This function uses vectorized operations to compute the peak-to-peak
        amplitude across the entire mesh, considering a specific number of
        stabilized cycles at the end of the simulation.

        Args:
            panel (Panel): The structural panel object.
            disp_history (ndarray): Displacement time history.
            last_cycles (int): Number of cycles from the end to use for
                               amplitude calculation. Defaults to 2.
            chunk_size (int): Number of mesh points to process per iteration.

        Returns:
            tuple: (max_value, max_xi, max_eta)
                - max_value: Max normalized peak-to-peak amplitude (w/h).
                - max_xi, max_eta: Natural coordinates of the max amplitude.
        """
        h = panel.laminate.total_thickness
        n_steps = disp_history.shape[1]

        # 1. Determine time window using a reference point (center of the mesh)
        ref_pt_idx = len(self.panel.mesh) // 2
        bdof_ref_w = self.panel.bdof[ref_pt_idx, 2, :]

        # Reconstruct history only for the reference point to detect cycles
        w_ref = (bdof_ref_w @ disp_history) / h

        zero_crossings = np.where(np.diff(np.sign(w_ref)))[0]
        needed_crossings = 2 * last_cycles

        if len(zero_crossings) > needed_crossings:
            t_start = zero_crossings[-(needed_crossings + 1)]
            t_end = zero_crossings[-1]
        else:
            # Fallback to the last 20% of simulation.
            t_start = int(0.8 * n_steps)
            t_end = n_steps

        # 2. Memory-efficient processing of amplitudes using chunks
        bdof_w = self.panel.bdof[:, 2, :]
        disp_window = disp_history[:, t_start:t_end]

        n_points = len(self.panel.mesh)
        max_w = np.full(n_points, -np.inf)
        min_w = np.full(n_points, np.inf)

        # Process points in batches to keep memory footprint low
        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)

            # Reconstruct 'w' only for the current batch of points
            w_chunk = (bdof_w[i:end_idx, :] @ disp_window) / h

            # Vectorized min/max search within the time window
            max_w[i:end_idx] = np.max(w_chunk, axis=1)
            min_w[i:end_idx] = np.min(w_chunk, axis=1)

        # 3. Calculate peak-to-peak amplitude
        amplitudes = (max_w - min_w) / 2.0
        max_idx = np.argmax(amplitudes)

        return (
            amplitudes[max_idx],
            self.panel.mesh[max_idx, 0],
            self.panel.mesh[max_idx, 1],
        )

    def get_point_amplitude(self, panel, disp_history, coords, last_cycles=2):
        """Calculate the LCO amplitude at specific natural coordinates.

        This function finds the closest mesh node to the provided (xi, eta)
        coordinates and reconstructs its displacement history. It is designed
        to be memory-efficient by only computing the 'w' component for that
        specific location.

        Args:
            panel (Panel): The structural panel object.
            disp_history (ndarray): Displacement time history.
            coords (tuple): Natural coordinates (xi, eta) of the target point.
            last_cycles (int): Number of cycles from the end to use for
                amplitude calculation. Defaults to 2.

        Returns:
            float: The normalized peak-to-peak amplitude (w/h).

        Raises:
            ValueError: If 'coords' is not a tuple/list of length 2.
        """
        h = panel.laminate.total_thickness
        n_steps = disp_history.shape[1]

        # 2. Reconstruct displacement history for the target point
        # Extracting only the 'w' component (DOF index 2)
        bdof_w_point = self.panel.eval_bdof_at_point(coords)
        w_point_full = (bdof_w_point @ disp_history)[0, 2, :] / h

        # 3. Identify stabilized cycle window
        # Zero-crossing detection on the local displacement signal
        zero_crossings = np.where(np.diff(np.sign(w_point_full)))[0]
        needed_crossings = 2 * last_cycles

        if len(zero_crossings) > needed_crossings:
            t_start = zero_crossings[-(needed_crossings + 1)]
            t_end = zero_crossings[-1]
        else:
            # Fallback to the last 20% of the simulation if cycles aren't clear
            t_start = int(0.8 * n_steps)
            t_end = n_steps

        # 4. Compute peak-to-peak amplitude in the window
        w_window = w_point_full[t_start:t_end]
        amplitude = (np.max(w_window) - np.min(w_window)) / 2.0

        return amplitude
