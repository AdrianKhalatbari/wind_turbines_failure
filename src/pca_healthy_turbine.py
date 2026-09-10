"""Initial PCA of the healthy wind-turbine SCADA data.

The analysis follows the course workflow: construct X, resolve essential
data issues, autoscale the healthy data, and compute PCA through SVD.
No response variable, fault label, or supervised method is used.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "resources" / "wind_turbine_fault_diagnosis_data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TURBINES = ["No.2WT", "No.14WT", "No.39WT"]
COMMON_VARIABLES = list(range(1, 28))
HEALTHY_TURBINE = "No.2WT"
LABEL_OFFSETS = {
    2: (10, 8), 3: (8, 18), 4: (8, 8), 5: (8, 20),
    6: (8, 7), 7: (8, -5), 8: (8, -17), 10: (8, 7),
    11: (8, -8), 13: (-16, 10), 14: (-16, -7), 20: (8, -29),
    21: (8, -18), 22: (8, 7), 23: (8, -12), 24: (8, -9),
}


def main() -> None:
    # Load the three structurally compatible turbines and retain variables
    # 1-27. This removes variable 28 from the healthy turbine.
    workbook = pd.read_excel(DATA_FILE, sheet_name=None, header=0)
    aligned = {
        name: workbook[name].loc[:, COMMON_VARIABLES].copy()
        for name in TURBINES
    }

    # The one missing value occurs in a single No.14WT row. Remove that row
    # rather than inventing a sensor value at an apparent operating transition.
    missing_rows = aligned["No.14WT"].index[aligned["No.14WT"].isna().any(axis=1)]
    aligned["No.14WT"] = aligned["No.14WT"].drop(index=missing_rows).reset_index(drop=True)

    # Variables 12 and 15 are constant in the healthy turbine. They cannot be
    # autoscaled and contain no variation for the healthy PCA model.
    healthy_aligned = aligned[HEALTHY_TURBINE]
    healthy_std = healthy_aligned.std(ddof=1)
    constant_variables = healthy_std.index[healthy_std == 0].tolist()
    pca_variables = [variable for variable in COMMON_VARIABLES if variable not in constant_variables]
    pca_x = {name: data.loc[:, pca_variables] for name, data in aligned.items()}
    healthy_x = pca_x[HEALTHY_TURBINE]

    # Fit pretreatment only on the healthy turbine. Any later turbine must use
    # these fixed means and scales; the PCA model must not be refitted.
    # Autoscaling follows the course guidance that variance acts as a weight.
    healthy_mean = healthy_x.mean()
    healthy_scale = healthy_x.std(ddof=1)
    healthy_x_scaled = (healthy_x - healthy_mean) / healthy_scale

    # Compute PCA through economy SVD: scores T = U*S and loadings P = V.
    u, singular_values, vt = np.linalg.svd(healthy_x_scaled.to_numpy(), full_matrices=False)
    scores = u * singular_values
    loadings = vt.T
    explained_variance = singular_values**2 / (len(healthy_x_scaled) - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    cumulative_ratio = np.cumsum(explained_ratio)

    save_numeric_outputs(
        pca_variables,
        healthy_mean,
        healthy_scale,
        scores,
        loadings,
        explained_variance,
        explained_ratio,
        cumulative_ratio,
    )
    create_pca_plots(healthy_x, pca_variables, scores, loadings, explained_ratio, cumulative_ratio)

    # Report the main pretreatment and PCA results needed for the next step.
    print("Aligned X matrices before PCA variable removal")
    for name, data in aligned.items():
        print(f"{name}: {data.shape[0]} observations x {data.shape[1]} variables")
    print(f"Removed No.14WT observation: {missing_rows[0] + 1}")
    print(f"Excluded zero-variance healthy variables: {constant_variables}")
    print("PCA-ready aligned X matrices")
    for name, data in pca_x.items():
        print(f"{name}: {data.shape[0]} observations x {data.shape[1]} variables")
    print(f"Final healthy PCA X: {healthy_x.shape[0]} observations x {healthy_x.shape[1]} variables")
    print("Pretreatment: mean-centering and unit-variance scaling using No.2WT statistics")
    variable_9 = healthy_x[9]
    print(
        "Variable 9 retained: it is not constant "
        f"({variable_9.nunique()} unique values, raw range "
        f"{variable_9.max() - variable_9.min():.3g}). Its low relative raw "
        "variation is influenced by its large offset; autoscaling gives it "
        "unit variance like the other retained variables."
    )

    print("\nExplained variance")
    for component in range(5):
        print(
            f"PC{component + 1}: {100 * explained_ratio[component]:.2f}% "
            f"(cumulative {100 * cumulative_ratio[component]:.2f}%)"
        )
    for threshold in (0.80, 0.90, 0.95):
        count = int(np.searchsorted(cumulative_ratio, threshold) + 1)
        print(f"Components for {int(threshold * 100)}% cumulative variance: {count}")

    print("\nVariables contributing most to the first three PCs")
    for component in range(3):
        order = np.argsort(np.abs(loadings[:, component]))[::-1][:6]
        values = ", ".join(
            f"{pca_variables[index]} ({loadings[index, component]:+.3f})"
            for index in order
        )
        print(f"PC{component + 1}: {values}")

    # Correlations support the interpretation of variables pointing in similar
    # or opposite directions in the loading plots and biplots.
    correlation = healthy_x.corr()
    pairs = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    print("\nStrongest positive healthy-variable correlations")
    for (first, second), value in pairs.sort_values(ascending=False).head(6).items():
        print(f"Variables {first} and {second}: r = {value:.3f}")
    print("Strongest negative healthy-variable correlations")
    for (first, second), value in pairs.sort_values().head(6).items():
        print(f"Variables {first} and {second}: r = {value:.3f}")


def save_numeric_outputs(
    variables: list[int],
    means: pd.Series,
    scales: pd.Series,
    scores: np.ndarray,
    loadings: np.ndarray,
    explained_variance: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_ratio: np.ndarray,
) -> None:
    """Save compact tables supporting interpretation and later projection."""
    component_names = [f"PC{i}" for i in range(1, len(variables) + 1)]

    pd.DataFrame(
        {
            "component": component_names,
            "explained_variance": explained_variance,
            "explained_variance_percent": 100 * explained_ratio,
            "cumulative_variance_percent": 100 * cumulative_ratio,
        }
    ).to_csv(OUTPUT_DIR / "pca_healthy_explained_variance.csv", index=False)

    loading_table = pd.DataFrame(loadings, index=variables, columns=component_names)
    loading_table.index.name = "variable"
    loading_table.to_csv(OUTPUT_DIR / "pca_healthy_loadings.csv")

    score_table = pd.DataFrame(scores[:, :5], columns=component_names[:5])
    score_table.index = np.arange(1, len(score_table) + 1)
    score_table.index.name = "observation"
    score_table.to_csv(OUTPUT_DIR / "pca_healthy_scores_first5.csv")

    pd.DataFrame(
        {"variable": variables, "healthy_mean": means, "healthy_std": scales}
    ).to_csv(OUTPUT_DIR / "pca_healthy_autoscaling_parameters.csv", index=False)


def create_pca_plots(
    healthy_x: pd.DataFrame,
    variables: list[int],
    scores: np.ndarray,
    loadings: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_ratio: np.ndarray,
) -> None:
    """Create the essential healthy-PCA figures used in the course workflow."""
    components = np.arange(1, len(explained_ratio) + 1)

    # Scree and cumulative explained-variance plot.
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.bar(components, 100 * explained_ratio, color="tab:blue", alpha=0.75,
             label="Individual variance")
    axis.plot(components, 100 * cumulative_ratio, color="tab:orange", marker="o",
              markersize=3, label="Cumulative variance")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance (%)")
    axis.set_xticks(components)
    axis.set_ylim(0, 103)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Healthy-turbine PCA explained variance")
    fig.savefig(OUTPUT_DIR / "pca_healthy_explained_variance.png", dpi=160)
    plt.close(fig)

    # Score plots for the first three PCs show the main healthy operating
    # trajectory. Colour represents observation order, not a response class.
    score_pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (x_pc, y_pc) in zip(axes, score_pairs):
        points = axis.scatter(
            scores[:, x_pc],
            scores[:, y_pc],
            c=np.arange(1, len(scores) + 1),
            cmap="viridis",
            s=11,
            alpha=0.6,
        )
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.set_xlabel(f"PC{x_pc + 1} ({100 * explained_ratio[x_pc]:.2f}%)")
        axis.set_ylabel(f"PC{y_pc + 1} ({100 * explained_ratio[y_pc]:.2f}%)")
        axis.set_title(f"PC{x_pc + 1}-PC{y_pc + 1}")
    fig.colorbar(points, ax=axes, label="Observation order", shrink=0.9)
    fig.suptitle("Healthy-turbine PCA scores")
    fig.savefig(OUTPUT_DIR / "pca_healthy_scores.png", dpi=160)
    plt.close(fig)

    # Loading positions indicate which variables vary together or oppositely.
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    axis.scatter(loadings[:, 0], loadings[:, 1], color="tab:red", s=35)
    for variable, x_value, y_value in zip(variables, loadings[:, 0], loadings[:, 1]):
        offset = LABEL_OFFSETS.get(variable, (4, 4))
        axis.annotate(str(variable), (x_value, y_value), xytext=offset,
                      textcoords="offset points", fontsize=9)
    axis.axhline(0, color="0.6", linewidth=0.8)
    axis.axvline(0, color="0.6", linewidth=0.8)
    axis.set_xlabel(f"PC1 loading ({100 * explained_ratio[0]:.2f}%)")
    axis.set_ylabel(f"PC2 loading ({100 * explained_ratio[1]:.2f}%)")
    axis.set_title("Healthy-turbine PCA loadings")
    axis.grid(alpha=0.2)
    fig.savefig(OUTPUT_DIR / "pca_healthy_loadings.png", dpi=160)
    plt.close(fig)

    # Signed loading bars identify the variables contributing most strongly to
    # each of the first three components.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for component, axis in enumerate(axes):
        order = np.argsort(np.abs(loadings[:, component]))[::-1][:8]
        order = order[np.argsort(loadings[order, component])]
        values = loadings[order, component]
        labels = [str(variables[index]) for index in order]
        colors = ["tab:orange" if value < 0 else "tab:blue" for value in values]
        axis.barh(labels, values, color=colors, alpha=0.8)
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_xlabel("Loading")
        axis.set_ylabel("Variable")
        axis.set_title(f"PC{component + 1} ({100 * explained_ratio[component]:.2f}%)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Largest healthy-turbine loading contributions")
    fig.savefig(OUTPUT_DIR / "pca_healthy_loading_contributions.png", dpi=160)
    plt.close(fig)

    # Biplots combine scores with loading directions. One common scale factor
    # is used for both arrow coordinates so their relative angles are preserved.
    # Correlations are interpreted primarily from the separate loading plot.
    biplot_pairs = [(0, 1), (0, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for axis, (x_pc, y_pc) in zip(axes, biplot_pairs):
        axis.scatter(scores[:, x_pc], scores[:, y_pc], color="tab:blue", s=10, alpha=0.2)
        score_extent = np.percentile(np.abs(scores[:, [x_pc, y_pc]]), 98, axis=0)
        loading_pair = loadings[:, [x_pc, y_pc]]
        component_limits = score_extent / np.max(np.abs(loading_pair), axis=0)
        arrow_scale = 0.78 * component_limits.min()
        if y_pc == 1:
            label_indices = set(
                np.argsort(np.linalg.norm(loading_pair, axis=1))[::-1][:12]
            )
        else:
            # In PC1-PC3, label only the strongest PC3 directions. Many PC1
            # loadings are almost equal and their labels would overlap.
            label_indices = set(np.argsort(np.abs(loadings[:, y_pc]))[::-1][:4])
        for index, (variable, loading) in enumerate(zip(variables, loading_pair)):
            end_x = loading[0] * arrow_scale
            end_y = loading[1] * arrow_scale
            axis.arrow(0, 0, end_x, end_y, color="tab:red", alpha=0.55,
                       width=0.01, head_width=0.15, length_includes_head=True)
            if index in label_indices:
                offset = LABEL_OFFSETS.get(variable, (4, 4))
                axis.annotate(str(variable), (end_x, end_y), xytext=offset,
                              textcoords="offset points", fontsize=8, color="tab:red")
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.set_xlabel(f"PC{x_pc + 1} scores ({100 * explained_ratio[x_pc]:.2f}%)")
        axis.set_ylabel(f"PC{y_pc + 1} scores ({100 * explained_ratio[y_pc]:.2f}%)")
        axis.set_title(f"PC{x_pc + 1}-PC{y_pc + 1}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    fig.suptitle("Healthy-turbine PCA biplots (loading arrows uniformly scaled)")
    fig.savefig(OUTPUT_DIR / "pca_healthy_biplot.png", dpi=160)
    plt.close(fig)

    # The correlation heatmap supports interpretation of loading directions.
    correlation = healthy_x.corr()
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(variables)), labels=variables, rotation=90)
    axis.set_yticks(np.arange(len(variables)), labels=variables)
    axis.set_xlabel("Variable")
    axis.set_ylabel("Variable")
    axis.set_title("Healthy-turbine variable correlations")
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    fig.savefig(OUTPUT_DIR / "pca_healthy_correlation_heatmap.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
