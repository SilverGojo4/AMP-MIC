# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals
"""
This module provides visualization utilities for Chaos Game Representation (CGR)
and Frequency Chaos Game Representation (FCGR) of antimicrobial peptide (AMP) sequences.

Main functionalities:
- `visualize_cgr_trajectory()`: Visualizes the CGR path overlaid on amino acid base structure.
- `visualize_fcgr_heatmap()`: Plots the FCGR matrix as a heatmap for a given sequence.

Both functions support white or transparent background, Times New Roman font styling,
and customization of resolution and color maps.
"""

# ============================== Standard Library Imports ==============================
import os
import sys
from typing import Union

import matplotlib.pyplot as plt

# ============================== Third-Party Library Imports ==============================
import numpy as np
import rpy2.robjects as ro
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Font Configuration ==============================
TIMES_NEW_ROMAN_BD = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"
)
TIMES_NEW_ROMAN = FontProperties(
    fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf"
)

# ============================== Project-Specific Imports ==============================
# CGR Encoding Utility Functions
from src.features.cgr_encoding import compute_cgr_features


# ============================== CGR Visualization Functions ==============================
def visualize_cgr_trajectory(
    cgr_result: ro.vectors.ListVector,
    output_file: str = None,  # type: ignore
    resolution: int = 16,
    transparent: bool = False,
):
    """
    Visualize CGR trajectory with amino acid base labels and connecting edges.

    Parameters
    ----------
    cgr_result : ro.vectors.ListVector
        R object returned from `kaos::cgr()`, containing base and trajectory.
    output_file : str, optional
        Path to save the output image. If None, the plot is only shown.
    resolution : int
        Grid resolution for the CGR coordinate system.
    transparent : bool
        Whether to render background as transparent.
    """
    custom_colors = {
        "Y": "#82B0D2",
        "A": "#82B0D2",
        "C": "#82B0D2",
        "D": "#82B0D2",
        "E": "#82B0D2",
        "F": "#FA7F6F",
        "G": "#FA7F6F",
        "H": "#FA7F6F",
        "I": "#FA7F6F",
        "K": "#FA7F6F",
        "L": "#8ECFC9",
        "M": "#8ECFC9",
        "N": "#8ECFC9",
        "P": "#8ECFC9",
        "Q": "#8ECFC9",
        "R": "#FF69B4",
        "S": "#FF69B4",
        "T": "#FF69B4",
        "V": "#FF69B4",
        "W": "#FF69B4",
    }

    base_x = np.array(cgr_result.rx2("base").rx2("x"))
    base_y = -np.array(cgr_result.rx2("base").rx2("y"))  # Flip Y-axis
    base_labels = np.array(ro.r["rownames"](cgr_result.rx2("base")))

    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    # Draw base edges
    shrink_ratio = 0.75
    for i in range(len(base_x)):
        x1, y1 = base_x[i], base_y[i]
        x2, y2 = base_x[(i + 1) % len(base_x)], base_y[(i + 1) % len(base_y)]
        plt.plot(
            [x1 + (x2 - x1) * shrink_ratio, x2 + (x1 - x2) * shrink_ratio],
            [y1 + (y2 - y1) * shrink_ratio, y2 + (y1 - y2) * shrink_ratio],
            color="black",
            linewidth=1.5,
            alpha=0.6,
            zorder=0,
        )

    # Grid
    ticks = np.linspace(-1, 1, resolution + 1)
    for t in ticks:
        plt.plot([t, t], [-1, 1], color="gray", linewidth=0.5)
        plt.plot([-1, 1], [t, t], color="gray", linewidth=0.5)

    # Axis arrows
    ax.annotate(
        "",
        xy=(1.2, 0),
        xytext=(-1.2, 0),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax.annotate(
        "",
        xy=(0, 1.2),
        xytext=(0, -1.2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

    # Base labels
    for label, x, y in zip(base_labels, base_x, base_y):
        color = custom_colors.get(label, "#000000")
        plt.text(
            x,
            y,
            label,
            fontsize=12,
            color=color,
            ha="center",
            va="center",
            fontproperties=TIMES_NEW_ROMAN_BD,
        )

    # Trajectory
    traj_x = np.array(cgr_result.rx2("x"))
    traj_y = -np.array(cgr_result.rx2("y"))
    plt.annotate(
        "",
        xy=(traj_x[0], traj_y[0]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.6),
    )
    for i in range(1, len(traj_x)):
        plt.annotate(
            "",
            xy=(traj_x[i], traj_y[i]),
            xytext=(traj_x[i - 1], traj_y[i - 1]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.6),
        )

    plt.scatter(traj_x, traj_y, color="black", s=15, alpha=0.9, zorder=3)

    # Final touches
    plt.xticks([]), plt.yticks([])  # type: ignore
    plt.xlim(-1.3, 1.3), plt.ylim(-1.3, 1.3)  # type: ignore
    ax.set_aspect("equal")
    plt.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    plt.close()


def visualize_fcgr_heatmap(
    cgr_result: ro.vectors.ListVector,
    cmap: Union[Colormap, str] = "Greys",
    output_file: str = None,  # type: ignore
    resolution: int = 16,
    transparent: bool = False,
):
    """
    Visualize FCGR heatmap for a single AMP sequence.

    Parameters
    ----------
    cgr_result : ro.vectors.ListVector
        R object returned from `kaos::cgr()`, containing base and matrix.
    cmap : Union[Colormap, str]
        Matplotlib colormap or custom colormap to apply (e.g., "hot", "Reds").
    output_file : str, optional
        Path to save the heatmap image. If None, the plot is only shown.
    resolution : int
        Resolution for the CGR matrix (e.g., 16 = 16x16).
    transparent : bool
        Whether to use transparent background.
    """
    x = np.array(cgr_result.rx2("x"))
    y = np.array(cgr_result.rx2("y"))

    def map_cgr_coords_to_fcgr_pixels(x_coords, y_coords, res):
        x_pixel = np.ceil((x_coords + 1) * res / 2).astype(int) - 1
        y_pixel = np.ceil((y_coords + 1) * res / 2).astype(int) - 1
        return y_pixel, x_pixel

    row_idx, col_idx = map_cgr_coords_to_fcgr_pixels(x, y, resolution)
    matrix = np.zeros((resolution, resolution), dtype=int)
    for r, c in zip(row_idx, col_idx):
        matrix[r, c] += 1

    # Plot
    plt.style.use("ggplot")
    plt.figure(figsize=(6, 5.5))
    ax = plt.gca()
    ax.set_facecolor("none" if transparent else "white")
    plt.gcf().set_facecolor("none" if transparent else "white")

    heatmap = sns.heatmap(
        matrix,
        cmap=cmap,
        square=True,
        cbar=True,
        linewidths=0.8,
        linecolor="lightgray",
        xticklabels=np.arange(resolution),  # type: ignore
        yticklabels=np.arange(resolution),  # type: ignore
        cbar_kws={"shrink": 0.7, "aspect": 20},
    )

    ax.tick_params(length=0)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=11, color="black"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), fontproperties=TIMES_NEW_ROMAN, fontsize=11, color="black"
    )

    cbar = heatmap.collections[0].colorbar
    for label in cbar.ax.get_yticklabels():  # type: ignore
        label.set_fontproperties(TIMES_NEW_ROMAN)
        label.set_color("black")

    plt.xlabel(
        "Width (Pixel)",
        fontsize=11,
        labelpad=10,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    plt.ylabel(
        "Height (Pixel)",
        fontsize=11,
        labelpad=10,
        fontproperties=TIMES_NEW_ROMAN_BD,
        color="black",
    )
    ax.set_aspect("equal")

    if transparent:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=transparent,
        )
    plt.close()


# Test script to visualize a single sequence
if __name__ == "__main__":

    # Test sequence
    # test_sequence = "FLPI"
    test_sequence = "FLPIVGKLLSGLSGLS"
    test_sequence = "GNTWE"

    # Heatmap bar
    custom_cmap = LinearSegmentedColormap.from_list(
        "my_red_white", ["#FFFFFF", "#8B0000"]
    )

    # Compute CGR features for the test sequence
    sequences = [test_sequence]
    cgr_vectors, cgr_results = compute_cgr_features(
        sequences=sequences,
        resolution=8,
    )
    visualize_cgr_trajectory(
        cgr_result=cgr_results[0],
        output_file=os.path.join(BASE_PATH, "outputs/materials/cgr_white.png"),
        resolution=8,
        transparent=False,
    )
    visualize_fcgr_heatmap(
        cgr_results[0],
        resolution=8,
        cmap=custom_cmap,
        output_file=os.path.join(BASE_PATH, "outputs/materials/fcgr_white.png"),
    )
