"""Inspect the supplied wind-turbine SCADA workbook.

This script reports raw-data challenges and creates a few simple raw-data
plots. It does not perform imputation, scaling, centering, PCA, or modelling.

Dataset context (from the project description): the SCADA measurements are
nominally recorded every 10 seconds and ordered sequentially. There is no
timestamp column, so continuity and synchronization between turbines cannot
be verified. WT2 is healthy and the other turbines develop faults. Variable
names and their physical meanings are not supplied.
"""

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# The script lives in src/, while the source workbook is kept in resources/.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "resources" / "wind_turbine_fault_diagnosis_data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PCA_CANDIDATES = ["No.2WT", "No.14WT", "No.39WT"]
COMMON_COLUMNS = list(range(1, 28))
PLOT_COLUMNS = [1, 5, 9, 11, 16]

# Nominal SCADA sampling interval, used only for reporting the time base.
SAMPLING_INTERVAL_SECONDS = 10

# Variables highlighted for cautious interpretation without altering them.
LOW_RELATIVE_VARIATION_VARIABLES = [9]
REGIME_VARIABLES = [5, 11, 16]


def main() -> None:
    # Load every workbook sheet. The first row contains numeric variable
    # identifiers, so keeping it as the header makes each later row an
    # observation and each column a predictor.
    datasets = pd.read_excel(DATA_FILE, sheet_name=None, header=0)

    print(f"Workbook: {DATA_FILE.name}")
    print(f"Sheets: {', '.join(datasets)}")
    print("\nDataset dimensions and structural checks")

    # Check dimensions, data types, numeric content, and missing cells.
    column_sets = {}
    for name, data in datasets.items():
        column_sets[name] = list(data.columns)
        numeric_cells = data.apply(lambda column: pd.to_numeric(column, errors="coerce")).notna()
        non_numeric_cells = int((~numeric_cells & data.notna()).sum().sum())
        missing_cells = int(data.isna().sum().sum())
        missing_columns = {
            str(column): int(count)
            for column, count in data.isna().sum().items()
            if count
        }

        print(
            f"{name}: observations={data.shape[0]}, variables={data.shape[1]}, "
            f"dtypes={data.dtypes.astype(str).value_counts().to_dict()}"
        )
        print(
            f"  all columns numeric: {non_numeric_cells == 0}; "
            f"missing cells: {missing_cells}"
        )
        if missing_columns:
            print(f"  columns with missing values: {missing_columns}")

    # Locate missing cells and print their nearby raw observations without
    # filling or otherwise changing the data.
    print("\nMissing-value inspection")
    for name, data in datasets.items():
        for row_index, column_index in zip(*data.isna().to_numpy().nonzero()):
            column = data.columns[column_index]
            print(
                f"{name}, column {column}: observation {row_index + 1} "
                f"(Excel row {row_index + 2})"
            )
            start = max(0, row_index - 2)
            stop = min(len(data), row_index + 3)
            left = max(0, column_index - 2)
            right = min(data.shape[1], column_index + 3)
            print(data.iloc[start:stop, left:right].to_string())

    # Save descriptive statistics for the three turbines being considered for
    # PCA. These are raw statistics, before any pretreatment.
    stats = []
    for name in PCA_CANDIDATES:
        summary = datasets[name].describe().T
        summary["range"] = summary["max"] - summary["min"]
        summary.insert(0, "turbine", name)
        summary.insert(1, "variable", summary.index)
        stats.append(summary.reset_index(drop=True))
    pd.concat(stats, ignore_index=True).to_csv(
        OUTPUT_DIR / "step2_raw_descriptive_statistics.csv", index=False
    )

    print("\nRaw scale and distribution checks for PCA candidates")
    for summary in stats:
        name = summary["turbine"].iloc[0]
        constant = summary.loc[summary["std"] == 0, "variable"].tolist()
        largest_ranges = summary.nlargest(3, "range")["variable"].tolist()
        print(
            f"{name}: constant variables={constant}; "
            f"largest raw ranges={largest_ranges}"
        )

    # Compare supplied column identifiers between turbine sheets.
    reference_name = next(iter(column_sets))
    reference_columns = column_sets[reference_name]
    print("\nColumn consistency")
    print(f"Reference sheet: {reference_name}")
    for name, columns in column_sets.items():
        missing_from_sheet = [column for column in reference_columns if column not in columns]
        extra_in_sheet = [column for column in columns if column not in reference_columns]
        print(
            f"{name}: same column labels as {reference_name}: "
            f"{not missing_from_sheet and not extra_in_sheet}"
        )
        if missing_from_sheet:
            print(f"  missing reference labels: {missing_from_sheet}")
        if extra_in_sheet:
            print(f"  extra labels: {extra_in_sheet}")

    # Record the current candidate set and common raw-variable structure.
    print("\nCandidate PCA data")
    print("Exclude No.3 because its 31-variable structure is incompatible with the other turbines.")
    print(f"Retain common variables: {COMMON_COLUMNS}")
    print("Remove variable 28 from No.2WT so all candidate X matrices have 27 columns.")
    print("No.14WT column 9 remains missing and must be handled before PCA.")

    # Explicit summary of the data challenges required by the assignment.
    # These are reported for the write-up only; no data is altered here.
    print("\nData challenges summary")

    # (a) The rows are sequential SCADA observations with a nominal 10-second
    # interval. Without timestamps, elapsed duration, gaps and synchronization
    # between turbines cannot be checked directly.
    for name in PCA_CANDIDATES:
        n_obs = datasets[name].shape[0]
        print(
            f"{name}: {n_obs} sequential observations, nominally recorded "
            f"every {SAMPLING_INTERVAL_SECONDS} s."
        )
    print("Measurement type: time series (sequential SCADA measurements).")
    print(
        "No explicit timestamp column is supplied. Recording gaps and "
        "synchronization between turbines therefore cannot be determined."
    )

    # (b) Variable 9 has a very large offset and little relative variation in
    # the healthy turbine. Its meaning is unknown, so it is retained rather
    # than labelled defective or removed.
    healthy = datasets[PCA_CANDIDATES[0]]
    for variable in LOW_RELATIVE_VARIATION_VARIABLES:
        mean = healthy[variable].mean()
        std = healthy[variable].std(ddof=1)
        value_range = healthy[variable].max() - healthy[variable].min()
        relative_std = std / abs(mean) if mean else float("inf")
        print(
            f"Variable {variable} in {PCA_CANDIDATES[0]} has a large offset "
            f"and low relative variation: mean={mean:.3g}, std={std:.3g}, "
            f"range={value_range:.3g}, relative std={relative_std:.2e}, "
            f"unique values={healthy[variable].nunique()}. It is retained, "
            "but its autoscaled PCA loading should be interpreted cautiously."
        )
    print(
        "Raw variable scales differ substantially. For example, variable 9 "
        "is approximately 10^7 in magnitude, while some variables have values "
        "near zero; this supports autoscaling before PCA."
    )

    # (c) The raw plots show several operating levels in variables 5, 11 and
    # 16. Simple maxima and upper quantiles also reveal isolated extremes in
    # No.14WT variables 5 and 11; these observations are only documented here.
    print(
        f"Variables with multiple levels or operating regimes in the raw "
        f"plots: {REGIME_VARIABLES}. Their unknown physical meanings prevent "
        "a more specific interpretation at this stage."
    )
    for variable in REGIME_VARIABLES:
        ranges = {
            name: float(datasets[name][variable].max() - datasets[name][variable].min())
            for name in PCA_CANDIDATES
            if variable in datasets[name].columns
        }
        formatted = ", ".join(f"{name}={value:.3g}" for name, value in ranges.items())
        print(f"  variable {variable} raw range per turbine: {formatted}")
    for variable in [5, 11]:
        data = datasets["No.14WT"][variable].dropna()
        print(
            f"  No.14WT variable {variable}: maximum={data.max():.3g}, "
            f"99th percentile={data.quantile(0.99):.3g}; the maximum is an "
            "obvious isolated extreme value."
        )

    # Create only the representative distribution and sequence plots needed
    # to understand the raw measurements before PCA.
    create_plots(datasets)

    print("\nImport convention: each sheet is an X matrix with observations in rows "
          "and supplied variable identifiers in columns.")


def create_plots(datasets: dict[str, pd.DataFrame]) -> None:
    """Create simple raw-data plots for the three PCA candidates."""
    colors = {"No.2WT": "tab:blue", "No.14WT": "tab:orange", "No.39WT": "tab:green"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, column in zip(axes.flat, PLOT_COLUMNS):
        for name in PCA_CANDIDATES:
            axis.hist(
                datasets[name][column].dropna(),
                bins=30,
                alpha=0.35,
                label=name,
                color=colors[name],
            )
        axis.set_title(f"Variable {column}")
        axis.set_xlabel("Raw value")
        axis.set_ylabel("Count")
    axes.flat[-1].axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Representative raw-variable distributions")
    fig.savefig(OUTPUT_DIR / "step2_raw_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(PLOT_COLUMNS), 1, figsize=(13, 10), constrained_layout=True
    )
    for axis, column in zip(axes, PLOT_COLUMNS):
        for name in PCA_CANDIDATES:
            axis.plot(
                datasets[name].index + 1,
                datasets[name][column],
                linewidth=0.8,
                label=name,
                color=colors[name],
            )
        axis.set_ylabel(f"Var {column}")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Observation order (nominal 10 s sampling)")
    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle("Selected raw variables over observation order")
    fig.savefig(OUTPUT_DIR / "step2_observation_order.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
