"""Time integration utilities for structural and aeroelastic analysis.

This module provides the Integrator class, which supports full- and
reduced-order dynamic simulations using different numerical schemes.
"""

import numpy as np
import scipy.linalg as la


class Integrator:
    """Time integration suite for structural and aeroelastic analysis.

    This class provides different numerical schemes (CDM, Generalized-Alpha)
    to solve non-linear dynamic systems in both Full Order (FOM) and
    Reduced Order (ROM) spaces.
    """

    def __init__(self, panel, tf, dt, m_mat, k_mat, c_mat, **kwargs):
        """Initialize simulation parameters and handle model reduction.

        By default, no modal reduction is performed (Full Order Model).

        Args:
            panel (Panel): The structural panel object.
            tf (float): Final simulation time [s].
            dt (float): Time step size [s].
            m_mat (ndarray): Mass matrix.
            k_mat (ndarray): Stiffness matrix.
            c_mat (ndarray): Damping matrix.
            **kwargs:
                n_modes (int): Number of modes for ROM. If not provided or -1,
                               the Full Order Model (FOM) is used.
                kind (str): Reduction basis type: 'NM' (Natural Modes) or
                           'AEM' (Aeroelastic Modes). Defaults to 'NM'.
        """
        self.panel = panel
        self.dt = dt
        self.tf = tf
        self.max_steps = int(tf / dt)
        self.time_array = np.linspace(0, tf, self.max_steps)

        # Determine if modal reduction is requested
        n_modes = kwargs.get("n_modes", -1)

        if n_modes > 0:
            # --- Reduced Order Model (ROM) Path ---
            kind = kwargs.get("kind", "NM")
            self._reduce_order(kind, n_modes, panel)

            # Update panel to compute non-linear forces directly in modal space
            panel.fnl_v(n_modes, self.h_mat, self.hl_mat)

            # Project global matrices to modal space: [M_m] = H_L.T @ M @ H_R
            self.m_mat = self.hl_mat.T @ m_mat @ self.h_mat
            self.k_mat = self.hl_mat.T @ k_mat @ self.h_mat
            self.c_mat = self.hl_mat.T @ c_mat @ self.h_mat
            self.size = n_modes
        else:
            # --- Full Order Model (FOM) Path (Default) ---
            self.m_mat = m_mat
            self.k_mat = k_mat
            self.c_mat = c_mat
            self.h_mat = np.eye(panel.len_tot)
            self.hl_mat = np.eye(panel.len_tot)
            self.size = panel.len_tot

        # Memory allocation for state vectors
        self.displ = np.zeros((self.size, self.max_steps))
        self.veloc = np.zeros((self.size, self.max_steps))
        self.accel = np.zeros((self.size, self.max_steps))
        self.f_ext = np.zeros((self.size, self.max_steps))

        self._set_initial_conditions()

    def _reduce_order(self, kind, n_modes):
        """Construct projection matrices for modal reduction.

        Args:
            kind (str): 'NM' for Normal Modes or 'AEM' for Aeroelastic Modes.
            n_modes (int): Number of modes to retain.
        """
        if kind == "NM":
            # Right and Left bases are identical for symmetric Normal Modes
            self.h_mat = np.real(self.panel.eigvecs[:, :n_modes])
            self.hl_mat = self.h_mat
        elif kind == "AEM":
            # Aeroelastic Modes: using bi-orthogonal basis (Left/Right)
            eigvals, eig_l, eig_r = la.eig(self.k_mat, self.m_mat, left=True)

            # Sort modes by frequency magnitude
            idx = np.abs(np.sqrt(eigvals)).argsort()
            self.h_mat = np.real(eig_r[:, idx[:n_modes]])
            self.hl_mat = np.real(eig_l[:, idx[:n_modes]])

    def _set_initial_conditions(self, **kwargs):
        """Set initial conditions based on a mode scale or a restart state.

        If not a restart, the initial condition is a fraction of the first
        vibration mode, scaled such that the displacement at point (0.5, 0)
        equals 1% of the panel thickness.

        Args:
            **kwargs:
                restart (bool): If True, uses the last state of the previous
                                simulation as the starting point.
        """
        restart = kwargs.get("restart", False)

        # Initialization od the variable with the initial conditions
        self.initial_condition = {
            "u0": None,
            "du0": None,
            "ddu0": None,
            "u_prev": None,
        }

        if restart:
            # Assumes the existence of a previous integration stored in the
            # panel or in a previous instance of this integrator.
            if hasattr(self, "last_state"):
                self.initial_condition["u0"] = self.last_state["u"]
                self.initial_condition["du0"] = self.last_state["du"]
                self.initial_condition["ddu0"] = self.last_state["ddu"]
                self.initial_condition["u_prev"] = self.last_state["u_prev"]
            else:
                raise ValueError("No previous state found for restart.")

        else:
            # --- Scaled First Mode Initialization ---
            # Reference point for scaling: (xi, eta) = (0.5, 0)
            # thickness (h) is retrieved from the laminate/panel properties
            h = self.panel.laminate.total_thickness
            target_disp = 0.01 * h

            # Get the first mode shape in physical space
            if not hasattr(self, "eigvecs"):
                self.panel.compute_free_modes()
            phi1 = np.real(self.panel.eigvecs[:, [0]])

            # Find the value of the first mode at point (0.5, 0)
            # This requires the basis functions (bdof) evaluated at the point
            point = np.array([[0.5, 0.0]])
            # w component
            w_at_point = self.panel.get_displacement_at_points(point, 2, phi1)

            # Calculate the scaling factor to meet 0.01 * h requirement
            scale_factor = target_disp / w_at_point
            u0_phys = phi1 * scale_factor
            du0_phys = np.zeros_like(u0_phys)

            # Project to modal space if ROM is active, otherwise assign to FOM
            if self.size < self.panel.len_tot:
                self.displ[:, [0]] = self.hl_mat.T @ (
                    self.panel.m_mat @ u0_phys
                )
                self.veloc[:, [0]] = self.hl_mat.T @ (
                    self.panel.m_mat @ du0_phys
                )
            else:
                self.displ[:, [0]] = u0_phys
                self.veloc[:, [0]] = du0_phys

            # Initial acceleration (a0) for the solver:
            # M*a0 = R_ext - K*u0 - C*v0
            u_init = self.displ[:, [0]]
            v_init = self.veloc[:, [0]]
            r_init = -self.panel.compute_nl(u_init)

            rhs_init = r_init - self.k_mat @ u_init - self.c_mat @ v_init
            a_init = la.solve(self.m_mat, rhs_init)
            self.accel[:, [0]] = a_init

            u_prev = u_init - self.dt * v_init + self.dt**2 / 2 * a_init
            self.initial_condition["u0"] = u_init
            self.initial_condition["du0"] = v_init
            self.initial_condition["ddu0"] = a_init
            self.initial_condition["u_prev"] = u_prev

    def save_restart_state(self):
        """Store the current state to allow a seamless restart later."""
        self.last_state = {
            "u": self.displ[:, [-1]],
            "du": self.veloc[:, [-1]],
            "ddu": self.accel[:, [-1]],
            "u_prev": self.displ[:, [-2]],
        }

    def solve_cdm(self, initial_weight=1.0, print_progress=False):
        """Perform time integration using the Central Difference Method (CDM).

        Args:
            initial_weight (float): Scale factor for the first mode as IC.
            print_progress (bool): If True, prints current step index.
        """
        dt = self.dt
        a0 = 1.0 / (dt**2)
        a1 = 1.0 / (2.0 * dt)
        a2 = 2.0 * a0

        # Effective mass matrix for CDM
        m_eff = a0 * self.m_mat + a1 * self.c_mat
        m_eff_inv = la.inv(m_eff)

        # Displacement at t = -dt to kickstart CDM
        u_prev = self.initial_condition["u_prev"]
        u_curr = self.initial_condition["u0"]

        for n in range(self.max_steps):
            # Compute non-linear internal forces
            r_curr = -self.panel.compute_nl(u_curr)

            # Equivalent force vector
            rhs = (
                r_curr
                - (self.k_mat - a2 * self.m_mat) @ u_curr
                - (a0 * self.m_mat - a1 * self.c_mat) @ u_prev
            )

            # Solve for next displacement
            u_next = m_eff_inv @ rhs

            # Reconstruct velocity and acceleration at time n
            self.accel[:, [n]] = a0 * (u_prev - 2.0 * u_curr + u_next)
            self.veloc[:, [n]] = a1 * (-u_prev + u_next)
            self.f_ext[:, [n]] = r_curr

            # Step forward
            u_prev = u_curr
            u_curr = u_next

            if n + 1 < self.max_steps:
                self.displ[:, [n + 1]] = u_next

            if print_progress and n % 500 == 0:
                print(f"Integration Step: {n} / {self.max_steps}")

    def solve_gen_alpha(self):
        """Generalized-Alpha Method placeholder."""
        print("Generalized-Alpha not yet implemented.")
        pass
