"""Verification for OpenPanelFlutter comparing results with Bardell (1992).

This script benchmark-tests free vibration frequency parameters for various
skew plates against literature results of Bardell (1992) available at
https://doi.org/10.1016/0045-7949(92)90044-Z.
"""

import logging
import sys
from pathlib import Path

import numpy as np

from openpanelflutter.definitions import (
    BasisFunction,
    BoundaryCondition,
    StructuralTheory,
)
from openpanelflutter.material import Isotropic, Laminate
from openpanelflutter.panel import Panel

logger = logging.getLogger("OpenPanelFlutter")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if the script is re-run in the same session
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

CURRENT_DIR = Path(__file__).resolve().parent

# --- Reference Databases from Bardell (1992) ---

OM_BA_SSSS = np.array(
    [
        [12.337, 19.739, 32.076, 41.946, 49.348, 49.348],
        [19.739, 49.348, 49.348, 78.957, 98.696, 98.696],
        [49.348, 78.957, 128.305, 167.783, 197.392, 197.392],
        [13.105, 20.652, 33.085, 44.745, 50.232, 52.486],
        [20.869, 48.205, 56.109, 79.044, 103.999, 108.888],
        [52.421, 82.609, 132.340, 178.981, 200.927, 209.945],
        [15.886, 23.923, 36.799, 52.638, 56.604, 63.200],
        [24.922, 52.638, 71.767, 83.840, 122.822, 122.822],
        [63.544, 95.691, 147.197, 210.552, 226.417, 252.801],
        [22.913, 32.017, 46.072, 63.475, 81.752, 82.885],
        [35.035, 66.278, 100.338, 107.658, 140.802, 168.290],
        [91.651, 128.068, 184.289, 253.901, 327.008, 331.541],
        [43.714, 55.300, 72.105, 92.379, 115.086, 139.902],
        [64.818, 104.955, 148.320, 196.294, 210.658, 249.254],
        [174.855, 221.200, 288.419, 369.516, 460.342, 559.606],
        [154.216, 173.448, 200.688, 229.787, 259.256, 294.428],
        [223.249, 283.133, 365.974, 438.848, 525.571, 609.182],
        [616.864, 693.792, 802.751, 919.150, 1037.02, 1177.71],
    ]
)

OM_BA_CCCC = np.array(
    [
        [24.577, 31.826, 44.770, 63.331, 63.983, 71.076],
        [35.985, 73.394, 73.394, 108.217, 131.581, 132.205],
        [98.311, 127.304, 179.079, 253.323, 255.933, 284.305],
        [26.222, 33.573, 46.633, 64.928, 68.698, 75.582],
        [38.187, 72.896, 82.618, 109.561, 138.974, 145.152],
        [104.887, 134.291, 186.533, 259.710, 274.793, 302.326],
        [32.192, 39.868, 53.366, 71.849, 85.081, 89.994],
        [46.090, 81.601, 105.167, 119.252, 164.986, 165.318],
        [128.767, 159.471, 213.462, 287.395, 340.324, 359.977],
        [47.324, 55.601, 70.063, 89.745, 112.134, 126.276],
        [65.643, 106.495, 148.312, 157.236, 196.773, 229.491],
        [189.295, 222.405, 280.252, 358.979, 448.536, 505.106],
        [92.428, 101.690, 117.873, 140.400, 167.162, 195.797],
        [121.647, 177.721, 231.751, 291.522, 304.805, 354.646],
        [369.713, 406.760, 471.492, 561.602, 668.647, 783.188],
        [337.567, 348.542, 367.481, 394.710, 429.900, 472.553],
        [407.657, 520.637, 619.938, 723.392, 827.945, 938.821],
        [1350.27, 1394.17, 1469.92, 1578.84, 1719.60, 1890.21],
    ]
)

OM_BA_CCSS = np.array(
    [
        [17.769, 25.198, 37.793, 52.342, 55.989, 59.586],
        [27.054, 60.539, 60.786, 92.836, 114.556, 114.704],
        [71.076, 100.792, 151.893, 209.369, 223.954, 238.343],
        [18.898, 26.436, 39.352, 55.597, 57.670, 63.324],
        [28.563, 59.927, 68.332, 93.566, 120.866, 125.956],
        [75.592, 105.744, 157.408, 222.386, 230.678, 253.297],
        [23.027, 30.934, 44.374, 62.054, 69.694, 75.975],
        [34.084, 66.343, 86.862, 100.638, 143.130, 143.296],
        [92.109, 123.737, 177.497, 248.217, 278.777, 303.901],
        [33.453, 42.026, 56.712, 75.866, 96.397, 103.116],
        [47.529, 85.187, 122.877, 129.008, 167.612, 197.695],
        [133.811, 168.103, 226.847, 303.465, 385.587, 412.463],
        [64.321, 73.805, 90.938, 113.960, 139.618, 165.885],
        [84.669, 138.529, 186.739, 241.393, 246.352, 299.020],
        [257.282, 295.220, 363.754, 455.841, 558.472, 663.540],
        [229.440, 240.637, 261.938, 292.873, 331.009, 372.680],
        [260.832, 381.096, 478.684, 567.632, 664.826, 761.713],
        [917.760, 1047.75, 1171.49, 1324.04, 1490.72, 1675.86],
    ]
)

OM_BA_SFSF = np.array(
    [
        [9.630, 16.135, 36.726, 38.945],
        [10.199, 16.487, 35.982, 41.439],
        [12.147, 17.703, 36.026, 49.394],
        [16.396, 20.386, 39.646, 59.624],
        [25.157, 26.458, 53.757, 72.165],
        [45.962, 46.905, 110.271, 114.023],
    ]
)

OM_BA_CFFF = np.array(
    [
        [3.471, 8.506, 21.284, 27.199],
        [3.583, 8.697, 22.230, 26.332],
        [3.929, 9.410, 25.290, 25.932],
        [4.508, 11.251, 26.973, 31.513],
        [5.251, 16.061, 30.415, 45.340],
        [6.064, 24.936, 49.383, 73.955],
    ]
)

DATABASE = [OM_BA_SSSS, OM_BA_CCCC, OM_BA_CCSS, OM_BA_SFSF, OM_BA_CFFF]

# --- Reference isotropic material properties for scaling ---
MAT_REF = Isotropic("MAT_REF", rho=76, E=130e6, nu=0.3)
LAMINATE = Laminate("Laminate")
LAMINATE.add_stack(MAT_REF, thicknesses=2e-3)


def _format_value(val):
    """Dynamic float string formatter matching Bardell's precision criteria.

    - Under 1000: 3 decimal places (e.g., 12.337, 200.688)
    - 1000 and above: 2 decimal places (e.g., 1177.71)
    """
    if abs(val) >= 1000.0:
        return f"{val:.2f}"
    return f"{val:.3f}"


def print_terminal_summary(
    boundary,
    basis_type,
    theory,
    n_func,
    n_gauss,
    data,
    ref,
    angles,
    aspects,
    is_cantilever=False,
):
    """Print a summary of the simulation configuration."""
    logger.info("\n" + "=" * 80)
    logger.info(" SIMULATION RUNTIME SETUP & CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f" {'Structural Theory':<25} : {theory.name}")
    logger.info(f" {'Boundary Conditions':<25} : {boundary}")
    logger.info(f" {'Basis Function Type':<25} : {basis_type.name}")
    logger.info(f" {'Number of Functions':<25} : {n_func}")
    logger.info(f" {'Gauss Quadrature Points':<25} : {n_gauss}\n")

    """Print an verification matrix block directly to the stdout terminal."""
    separator_line = "=" * 76
    sub_separator = "-" * 76

    logger.info("\n" + separator_line)
    logger.info(f" BENCHMARK SUMMARY: BOUNDARY CONDITION [{boundary}]")
    logger.info(separator_line)
    logger.info(
        f"{'Skew (deg)':<10} | {'Aspect (a/b)':<12} | {'Mode':<5} |"
        f" {'Calculated':<12} | {'Reference':<12} | {'Error':<9}"
    )
    logger.info(sub_separator)

    for ii in range(ref.shape[0]):
        # Dynamically evaluate geometric parameters mapping the database row
        if is_cantilever:
            skew_val = angles[ii]
            aspect_val = 1.0
        else:
            skew_val = angles[ii // 3]
            aspect_val = aspects[ii % 3]

        for jj in range(ref.shape[1]):
            dd = data[ii, jj]
            rr = ref[ii, jj]
            diff_perc = 100 * (dd - rr) / rr
            if f"{diff_perc:.2f}" == "-0.00":
                diff_perc = 0.00

            # Geometry headers are only printed on the first mode row block
            # to preserve scannability
            skew_str = f"{skew_val:.1f}°" if jj == 0 else ""
            aspect_str = f"{aspect_val:.1f}" if jj == 0 else ""

            logger.info(
                f"{skew_str:<10} | {aspect_str:<12} | M{jj + 1:<4} |"
                f" {_format_value(dd):>12} | {_format_value(rr):>12} |"
                f" {diff_perc:>7.2f}%"
            )
        if ii < ref.shape[0] - 1:
            logger.info(sub_separator)
    logger.info(separator_line + "\n")


def write_comp_table(
    data, ref, angles, aspects, boundary, theory, percent=True
):
    """Generate LaTeX comparative tables for 6-mode configurations."""
    str_perc = "_perc" if percent else ""
    output_dir = CURRENT_DIR / "tables" / "compare_Bardell"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"comp{boundary}{theory}{str_perc}.tex"

    with open(file_path, "w", encoding="utf-8") as file_tex:
        file_tex.write(r"\begin{tabular}{cc|c|c|c|c|c|c}" + "\n")
        file_tex.write(r"\toprule" + "\n")

        header = (
            r" \multirow{3}{*}{\gls{beta}} & "
            r"\multirow{3}{*}{\glslink{ab}{$a/b$}} & "
            r"\multicolumn{1}{c|}{Mode 1} & \multicolumn{1}{c|}{Mode 2} & "
            r"\multicolumn{1}{c|}{Mode 3} & \multicolumn{1}{c|}{Mode 4} & "
            r"\multicolumn{1}{c|}{Mode 5} & \multicolumn{1}{c}{Mode 6}"
            r" \\ \cline{3-8}" + "\n"
        )
        file_tex.write(header)

        header_sub1 = (
            r" & & \gls{Omega} & \gls{Omega} & \gls{Omega} & \gls{Omega} &"
            r"\gls{Omega} & \gls{Omega} \\" + "\n"
        )
        header_sub2 = (
            r" &  & Dif [$\%$] & Dif [$\%$] & Dif [$\%$] & Dif [$\%$] & "
            r"Dif [$\%$] & Dif [$\%$] \\" + "\n"
            if percent
            else r" &  & \gls{Omega} Ref & \gls{Omega} Ref & \gls{Omega} Ref &"
            r" \gls{Omega} Ref & \gls{Omega} Ref & \gls{Omega} Ref \\" + "\n"
        )

        file_tex.write(header_sub1)
        file_tex.write(header_sub2)
        file_tex.write(r"\midrule \midrule" + "\n")

        angle_idx = 0
        for ii in range(ref.shape[0]):
            if ii % 3 == 0:
                if angle_idx > 0:
                    file_tex.write(r"\midrule" + "\n")
                row_start = (
                    rf"\multirow{{6}}{{*}}{{${angles[angle_idx]:.0f}^\circ$}} "
                )
                row_start += r" & "

                if aspects[ii % 3] in [1.0, 2.0]:
                    row_start += (
                        rf"\multirow{{2}}{{*}}{{${aspects[ii % 3]:.0f}$}} "
                    )
                else:
                    row_start += (
                        rf"\multirow{{2}}{{*}}{{${aspects[ii % 3]:.1f}$}} "
                    )
                angle_idx += 1
            else:
                if aspects[ii % 3] in [1.0, 2.0]:
                    row_start = (
                        rf" & \multirow{{2}}{{*}}{{${aspects[ii % 3]:.0f}$}} "
                    )
                else:
                    row_start = (
                        rf" & \multirow{{2}}{{*}}{{${aspects[ii % 3]:.1f}$}} "
                    )

            row_alt = " & "
            for jj in range(ref.shape[1]):
                dd = data[ii, jj]
                rr = ref[ii, jj]
                diff_perc = 100 * (dd - rr) / rr
                if f"{diff_perc:.2f}" == "-0.00":
                    diff_perc = 0.00

                # Apply dynamic precision format
                row_start += f" & ${_format_value(dd)}$ "
                if percent:
                    row_alt += f"& {diff_perc:.2f} "
                else:
                    row_alt += f"& ${_format_value(rr)}$ "

            row_start += r"\\" + "\n"
            row_alt += (
                r" \\ \cline{3-8}" + "\n"
                if (ii <= ref.shape[0] - 2 and ii % 3 < 2)
                else r"\\" + "\n"
            )

            file_tex.write(row_start)
            file_tex.write(row_alt)

        file_tex.write(r"\bottomrule" + "\n")
        file_tex.write(r"\end{tabular}")


def write_comp_table_4mode(data, ref, angles, boundary, theory, percent=True):
    """Generate LaTeX comparative tables for 4-mode configurations."""
    str_perc = "_perc" if percent else ""
    output_dir = CURRENT_DIR / "tables" / "compare_Bardell"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"comp{boundary}{theory}{str_perc}.tex"

    with open(file_path, "w", encoding="utf-8") as file_tex:
        file_tex.write(r"\begin{tabular}{c|c|c|c|c}" + "\n")
        file_tex.write(r"\toprule" + "\n")

        header = (
            r" \multirow{3}{*}{\gls{beta}} & "
            r"\multicolumn{1}{c|}{Mode 1} & \multicolumn{1}{c|}{Mode 2} & "
            r"\multicolumn{1}{c|}{Mode 3} & \multicolumn{1}{c}{Mode 4}"
            r" \\ \cline{2-5}" + "\n"
        )
        file_tex.write(header)

        header_sub1 = (
            r" & \gls{Omega} & \gls{Omega} & \gls{Omega} & \gls{Omega} \\"
            + "\n"
        )
        header_sub2 = (
            r"& Dif [$\%$] & Dif [$\%$] & Dif [$\%$] & Dif [$\%$]\\" + "\n"
            if percent
            else r" & \gls{Omega} Ref & \gls{Omega} Ref & \gls{Omega} Ref & "
            r"\gls{Omega} Ref \\" + "\n"
        )

        file_tex.write(header_sub1)
        file_tex.write(header_sub2)
        file_tex.write(r"\midrule \midrule" + "\n")

        angle_idx = 0
        for ii in range(ref.shape[0]):
            if angle_idx > 0:
                file_tex.write(r"\midrule" + "\n")
            row_start = (
                rf"\multirow{{2}}{{*}}{{${angles[angle_idx]:.0f}^\circ$}} "
            )
            angle_idx += 1
            row_alt = " "

            for jj in range(ref.shape[1]):
                dd = data[ii, jj]
                rr = ref[ii, jj]
                diff_perc = 100 * (dd - rr) / rr
                if f"{diff_perc:.2f}" == "-0.00":
                    diff_perc = 0.00

                row_start += f" & ${_format_value(dd)}$ "
                if percent:
                    row_alt += f"& {diff_perc:.2f} "
                else:
                    row_alt += f"& ${_format_value(rr)}$ "

            row_start += r"\\" + "\n"
            row_alt += r"\\" + "\n"
            file_tex.write(row_start)
            file_tex.write(row_alt)

        file_tex.write(r"\bottomrule" + "\n")
        file_tex.write(r"\end{tabular}")


def run_case(boundary, n_func, save_tables=False, percent_table=False):
    """Run verification and generate benchmarking data against literature."""
    angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0])
    aspects = np.array([0.5, 1.0, 2.0])

    bound_str = boundary.value

    if bound_str == "SSSS":
        reference = DATABASE[0]
    elif bound_str == "CCCC":
        reference = DATABASE[1]
    elif bound_str == "CCSS":
        reference = DATABASE[2]
    elif bound_str == "SFSF":
        reference = DATABASE[3]
    elif bound_str == "CFFF":
        reference = DATABASE[4]

    ans_rm = np.zeros(shape=reference.shape)
    row_counter = 0
    is_4modes = True if bound_str == "SFSF" or bound_str == "CFFF" else False

    for skew_ang in angles:
        # 4-mode cases (SFSF, CFFF), slice only the relevant aspect ratios
        current_aspects = [1.0] if is_4modes else aspects

        for asp in current_aspects:
            print(
                f"Running: Boundary={bound_str} | NFUNC={n_func} |"
                f" beta={skew_ang:.0f}° | Aspect={asp}"
            )

            # Execute solver instance
            panel = Panel(
                length=700e-3,
                width=700e-3 / asp,
                laminate=LAMINATE,
                beta=skew_ang,
            )
            panel.setup_kinematics(
                n_func,
                theory=StructuralTheory.REISSNER_MINDLIN,
                basis_type=BasisFunction.BARDELL,
                boundary_conditions=boundary,
            )
            panel.compute_free_modes()

            material = panel.laminate.plies[0].material
            h = panel.laminate.total_thickness
            e = material.E
            rho = material.rho
            nu = material.nu
            a = panel.a

            # Structural parameter scaling calculations
            d_flexural_rigidity = (e * h**3) / (1 - nu**2) / 12
            omega_scale_factor = (
                rho * h * (a**4) / d_flexural_rigidity
            ) ** 0.5
            omega_rm_scaled = (
                omega_scale_factor * 2 * np.pi * panel.free_omega_hz
            )

            # Extract first active frequencies matching benchmark limits
            num_modes = reference.shape[1]
            ans_rm[row_counter, :] = omega_rm_scaled[0:num_modes]
            row_counter += 1

    # Print detailed matrix summary directly to terminal screen
    print_terminal_summary(
        bound_str,
        panel.basis_type,
        panel.theory,
        n_func,
        panel.n_gauss,
        ans_rm,
        reference,
        angles,
        aspects,
        is_4modes,
    )

    if save_tables:
        # Output table files for LaTeX generation, with both percentage
        # error and absolute reference values
        if is_4modes:
            write_comp_table_4mode(
                ans_rm,
                reference,
                angles,
                bound_str,
                panel.theory.value,
                percent=percent_table,
            )
        else:
            write_comp_table(
                ans_rm,
                reference,
                angles,
                aspects,
                bound_str,
                panel.theory.value,
                percent=percent_table,
            )


if __name__ == "__main__":
    # logger setup for file output
    output_dir = CURRENT_DIR / "tables" / "compare_Bardell"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = output_dir / "benchmark_run.log"
    file_handler = logging.FileHandler(
        log_filepath, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    # Execute all benchmark cases and generate comparative tables for LaTeX
    # n_func are selected according to Bardell for each boundary condition.
    run_case(BoundaryCondition.SSSS, n_func=18, save_tables=True)

    run_case(BoundaryCondition.CCCC, n_func=16, save_tables=True)

    run_case(BoundaryCondition.CCSS, n_func=17, save_tables=True)

    run_case(BoundaryCondition.SFSF, n_func=19, save_tables=True)

    run_case(BoundaryCondition.CFFF, n_func=19, save_tables=True)
