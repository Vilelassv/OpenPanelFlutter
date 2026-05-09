"""Boundary Conditions Module - Bardell Polynomials Indices.

This module provides Bardell polynomials indices to define boundary
conditions based on the structural theory and boundary type.
"""

from .definitions import BoundaryCondition, StructuralTheory


def get_bardell_indices(theory, bound, nfunc):
    """Get the indices for Bardell polynomials based on BC and theory.

    This function explicitly defines the polynomial indices for each field
    (u, v, w, beta_x, beta_y).

    Note: 'm' represents the xi direction and 'n' represents the eta direction.

    In Reissner-Mindlin theory, 'w' indices greater than 4 are incremented
    by one to maintain kinematic consistency and mitigate shear locking
    while preserving the total number of shape functions (nfunc).

    Args:
        theory (StructuralTheory): The structural theory (RM or Kirchhoff).
        bound (BoundaryCondition): The boundary condition type.
        nfunc (int): Number of shape functions to be used.

    Returns:
        list: [m_u, n_u, m_v, n_v, m_w, n_w, m_bx, n_bx, m_by, n_by]
    """
    # --- REISSNER-MINDLIN THEORY ---
    if theory == StructuralTheory.REISSNER_MINDLIN:
        if bound == BoundaryCondition.CCCC:
            # Membrane and rotation indices (m: xi direction, n: eta direction)
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))
            m_bx = n_bx = m_by = n_by = [2, 4] + list(range(5, nfunc + 3))
            # x indices: incremented > 4 to avoid shear locking
            m_w = n_w = [2, 4] + list(range(6, nfunc + 4))

        elif bound == BoundaryCondition.SSSS:
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))
            m_w = n_w = [2, 4] + list(range(6, nfunc + 4))
            m_bx = list(range(1, nfunc + 1))
            n_bx = [2, 4] + list(range(5, nfunc + 3))
            m_by = [2, 4] + list(range(5, nfunc + 3))
            n_by = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.SSSSsoft:
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))
            m_w = n_w = [2, 4] + list(range(6, nfunc + 4))
            m_bx = n_bx = m_by = n_by = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.CCSS:
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))
            m_w = n_w = [2, 4] + list(range(6, nfunc + 4))
            m_bx = n_bx = m_by = n_by = [2, 3, 4] + list(range(5, nfunc + 2))

        elif bound == BoundaryCondition.SFSF:
            m_u = m_v = [2, 4] + list(range(5, nfunc + 3))
            n_u = n_v = list(range(1, nfunc + 1))
            m_w = [2, 4] + list(range(6, nfunc + 4))
            n_w = list(range(1, nfunc + 1))
            m_bx = n_bx = m_by = n_by = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.CFCF:
            m_u = m_v = [2, 4] + list(range(5, nfunc + 3))
            n_u = n_v = list(range(1, nfunc + 1))
            m_w = [2, 4] + list(range(6, nfunc + 4))
            n_w = list(range(1, nfunc + 1))
            m_bx = m_by = [2, 4] + list(range(5, nfunc + 3))
            n_bx = n_by = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.CFFF:
            m_u = m_v = [2, 3, 4] + list(range(5, nfunc + 2))
            n_u = n_v = list(range(1, nfunc + 1))
            m_w = [2, 3, 4] + list(range(6, nfunc + 3))
            n_w = list(range(1, nfunc + 1))
            m_bx = m_by = [2, 3, 4] + list(range(5, nfunc + 2))
            n_bx = n_by = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.FCFF:
            m_u = m_v = list(range(1, nfunc + 1))
            n_u = n_v = [2, 3, 4] + list(range(5, nfunc + 2))
            m_w = list(range(1, nfunc + 1))
            n_w = [2, 3, 4] + list(range(6, nfunc + 3))
            m_bx = m_by = list(range(1, nfunc + 1))
            n_bx = n_by = [2, 3, 4] + list(range(5, nfunc + 2))

    # --- KIRCHHOFF THEORY ---
    elif theory == StructuralTheory.KIRCHHOFF:
        # Rotations are not independent variables in Kirchhoff theory
        m_bx = n_bx = m_by = n_by = []

        if bound == BoundaryCondition.CCCC:
            m_w = n_w = list(range(5, nfunc + 5))
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))

        elif bound in [BoundaryCondition.SSSS, BoundaryCondition.SSSSsoft]:
            m_w = n_w = [2, 4] + list(range(5, nfunc + 3))
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))

        elif bound == BoundaryCondition.CCSS:
            m_w = n_w = [4] + list(range(5, nfunc + 4))
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))

        elif bound == BoundaryCondition.CFFF:
            m_w = [3, 4] + list(range(5, nfunc + 3))
            n_w = list(range(1, nfunc + 1))
            m_u = m_v = [2, 3, 4] + list(range(5, nfunc + 2))
            n_u = n_v = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.SFSF:
            m_w = [2, 4] + list(range(5, nfunc + 3))
            n_w = list(range(1, nfunc + 1))
            m_u = m_v = [2, 4] + list(range(5, nfunc + 3))
            n_u = n_v = list(range(1, nfunc + 1))

        elif bound == BoundaryCondition.CSCS:
            m_w = list(range(5, nfunc + 5))
            n_w = [2, 4] + list(range(5, nfunc + 3))
            m_u = n_u = m_v = n_v = [2, 4] + list(range(5, nfunc + 3))

        elif bound == BoundaryCondition.FCFF:
            m_w = list(range(1, nfunc + 1))
            n_w = [3, 4] + list(range(5, nfunc + 3))
            m_u = m_v = list(range(1, nfunc + 1))
            n_u = n_v = [2, 3, 4] + list(range(5, nfunc + 2))

    return [m_u, n_u, m_v, n_v, m_w, n_w, m_bx, n_bx, m_by, n_by]
