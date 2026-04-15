#!/usr/bin/env python3
"""
Render the YOLO-DPAR "Results and weights" table as a styled PNG image.

Output:
    images/results_weights_table.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _build_dataframe() -> pd.DataFrame:
    rows = [
        {
            "Model": "yolo11l-v10r-210825",
            "End2End": "No",
            "Params (M)": 26.4,
            "GFLOPs": 93.9,
            "Person": 0.885,
            "Face": 0.854,
            "Vehicle": 0.750,
            "PoseL": 0.873,
            "FaceKPL": 0.775,
            "AttrMain": 0.493,
            "ClrTop": 0.561,
            "ClrBot": 0.391,
            "Weapon": 0.808,
            "Threat": 0.643,
            "FIQA": 0.644,
        },
        {
            "Model": "yolo26l-v10-240226",
            "End2End": "No",
            # Per request: clone non-ReID metrics from yolo26l-v10r-240226.
            "Params (M)": 26.5,
            "GFLOPs": 94.3,
            "Person": 0.884,
            "Face": 0.864,
            "Vehicle": 0.746,
            "PoseL": 0.890,
            "FaceKPL": 0.771,
            "AttrMain": 0.512,
            "ClrTop": 0.580,
            "ClrBot": 0.402,
            "Weapon": 0.841,
            "Threat": 0.680,
            "FIQA": 0.653,
        },
        {
            "Model": "yolo26l-v10r-240226",
            "End2End": "No",
            "Params (M)": 26.7,
            "GFLOPs": 96.7,
            "Person": 0.884,
            "Face": 0.864,
            "Vehicle": 0.746,
            "PoseL": 0.890,
            "FaceKPL": 0.771,
            "AttrMain": 0.512,
            "ClrTop": 0.580,
            "ClrBot": 0.402,
            "Weapon": 0.841,
            "Threat": 0.680,
            "FIQA": 0.653,
        },
        {
            "Model": "yolo26l-e2e-v10r-080426",
            "End2End": "Yes",
            "Params (M)": 29.7,
            "GFLOPs": 113.0,
            "Person": 0.879,
            "Face": 0.878,
            "Vehicle": 0.743,
            "PoseL": 0.886,
            "FaceKPL": 0.771,
            "AttrMain": 0.498,
            "ClrTop": 0.578,
            "ClrBot": 0.403,
            "Weapon": 0.825,
            "Threat": 0.678,
            "FIQA": 0.638,
        },
        {
            "Model": "yolo26s-v10-030426-80",
            "End2End": "No",
            "Params (M)": 10.8,
            "GFLOPs": 25.7,
            "Person": 0.873,
            "Face": 0.858,
            "Vehicle": 0.700,
            "PoseL": 0.876,
            "FaceKPL": 0.765,
            "AttrMain": 0.403,
            "ClrTop": 0.477,
            "ClrBot": 0.359,
            "Weapon": 0.765,
            "Threat": 0.611,
            "FIQA": 0.599,
        },
        {
            "Model": "yolo26s-v10-210226",
            "End2End": "No",
            "Params (M)": 10.8,
            "GFLOPs": 25.7,
            "Person": 0.868,
            "Face": 0.836,
            "Vehicle": 0.674,
            "PoseL": 0.873,
            "FaceKPL": 0.745,
            "AttrMain": 0.473,
            "ClrTop": 0.562,
            "ClrBot": 0.391,
            "Weapon": 0.814,
            "Threat": 0.651,
            "FIQA": 0.645,
        },
        {
            "Model": "yolo26s-e2e-v10-100426",
            "End2End": "Yes",
            "Params (M)": 12.5,
            "GFLOPs": 33.2,
            "Person": 0.861,
            "Face": 0.854,
            "Vehicle": 0.693,
            "PoseL": 0.859,
            "FaceKPL": 0.753,
            "AttrMain": 0.458,
            "ClrTop": 0.549,
            "ClrBot": 0.383,
            "Weapon": 0.790,
            "Threat": 0.630,
            "FIQA": 0.625,
        },
        {
            "Model": "yolo11s-v10-210825",
            "End2End": "No",
            "Params (M)": 10.1,
            "GFLOPs": 24.1,
            "Person": 0.866,
            "Face": 0.822,
            "Vehicle": 0.684,
            "PoseL": 0.839,
            "FaceKPL": 0.754,
            "AttrMain": 0.455,
            "ClrTop": 0.552,
            "ClrBot": 0.377,
            "Weapon": 0.784,
            "Threat": 0.606,
            "FIQA": 0.640,
        },
        {
            "Model": "yolo26n-v10-030426",
            "End2End": "No",
            "Params (M)": 3.26,
            "GFLOPs": 8.76,
            "Person": 0.824,
            "Face": 0.770,
            "Vehicle": 0.498,
            "PoseL": 0.824,
            "FaceKPL": 0.722,
            "AttrMain": 0.423,
            "ClrTop": 0.520,
            "ClrBot": 0.363,
            "Weapon": 0.739,
            "Threat": 0.551,
            "FIQA": 0.636,
        },
        {
            "Model": "yolo26n-e2e-v10-050426",
            "End2End": "Yes",
            "Params (M)": 4.2,
            "GFLOPs": 12.9,
            "Person": 0.813,
            "Face": 0.780,
            "Vehicle": 0.547,
            "PoseL": 0.785,
            "FaceKPL": 0.680,
            "AttrMain": 0.366,
            "ClrTop": 0.481,
            "ClrBot": 0.336,
            "Weapon": 0.679,
            "Threat": 0.509,
            "FIQA": 0.594,
        },
        {
            "Model": "yolo11l (stock)",
            "End2End": "No",
            "Params (M)": 25.4,
            "GFLOPs": 87.6,
            "Person": 0.804,
            "Face": np.nan,
            "Vehicle": 0.724,
            "PoseL": np.nan,
            "FaceKPL": np.nan,
            "AttrMain": np.nan,
            "ClrTop": np.nan,
            "ClrBot": np.nan,
            "Weapon": np.nan,
            "Threat": np.nan,
            "FIQA": np.nan,
        },
        {
            "Model": "yolo26l (stock)",
            "End2End": "No",
            "Params (M)": 26.3,
            "GFLOPs": 93.8,
            "Person": 0.804,
            "Face": np.nan,
            "Vehicle": 0.737,
            "PoseL": np.nan,
            "FaceKPL": np.nan,
            "AttrMain": np.nan,
            "ClrTop": np.nan,
            "ClrBot": np.nan,
            "Weapon": np.nan,
            "Threat": np.nan,
            "FIQA": np.nan,
        },
        {
            "Model": "yolo11s (stock)",
            "End2End": "No",
            "Params (M)": 9.46,
            "GFLOPs": 21.7,
            "Person": 0.770,
            "Face": np.nan,
            "Vehicle": 0.655,
            "PoseL": np.nan,
            "FaceKPL": np.nan,
            "AttrMain": np.nan,
            "ClrTop": np.nan,
            "ClrBot": np.nan,
            "Weapon": np.nan,
            "Threat": np.nan,
            "FIQA": np.nan,
        },
        {
            "Model": "yolo26s (stock)",
            "End2End": "No",
            "Params (M)": 10.0,
            "GFLOPs": 22.8,
            "Person": 0.765,
            "Face": np.nan,
            "Vehicle": 0.677,
            "PoseL": np.nan,
            "FaceKPL": np.nan,
            "AttrMain": np.nan,
            "ClrTop": np.nan,
            "ClrBot": np.nan,
            "Weapon": np.nan,
            "Threat": np.nan,
            "FIQA": np.nan,
        },
        {
            "Model": "yolo26n (stock)",
            "End2End": "No",
            "Params (M)": 2.57,
            "GFLOPs": 6.12,
            "Person": 0.696,
            "Face": np.nan,
            "Vehicle": 0.580,
            "PoseL": np.nan,
            "FaceKPL": np.nan,
            "AttrMain": np.nan,
            "ClrTop": np.nan,
            "ClrBot": np.nan,
            "Weapon": np.nan,
            "Threat": np.nan,
            "FIQA": np.nan,
        },
    ]

    # Full ReID evaluation table payload (kept in code for reproducibility).
    reid_eval_rows = [
        {
            "dataset": "Mean",
            "test": "yolo26l-e2e-v10r-080426.pt",
            "grid": "mean",
            "Num id": 526,
            "Num Img": 7930,
            "Missed": 24.2,
            "Emb": 7900,
            "Q total": 526,
            "Q valid": 526,
            "Rank-1": 0.585,
            "Rank-5": 0.752,
            "Rank-10": 0.809,
            "Rank-20": 0.859,
            "mAP": 0.354,
            "Vis": 4.75,
        },
        {
            "dataset": "Mean",
            "test": "yolo26l-v10r-240226.pt",
            "grid": "mean",
            "Num id": 526,
            "Num Img": 7930,
            "Missed": 27.2,
            "Emb": 7900,
            "Q total": 526,
            "Q valid": 526,
            "Rank-1": 0.516,
            "Rank-5": 0.711,
            "Rank-10": 0.772,
            "Rank-20": 0.827,
            "mAP": 0.291,
            "Vis": 4.75,
        },
        {
            "dataset": "Mean",
            "test": "yolo11l.pt",
            "grid": "mean",
            "Num id": 526,
            "Num Img": 7930,
            "Missed": 57.2,
            "Emb": 7870,
            "Q total": 526,
            "Q valid": 526,
            "Rank-1": 0.033,
            "Rank-5": 0.071,
            "Rank-10": 0.097,
            "Rank-20": 0.144,
            "mAP": np.nan,
            "Vis": 4.75,
        },
        {
            "dataset": "Mean",
            "test": "yolo26l.pt",
            "grid": "mean",
            "Num id": 526,
            "Num Img": 7930,
            "Missed": 36.4,
            "Emb": 7890,
            "Q total": 526,
            "Q valid": 526,
            "Rank-1": 0.032,
            "Rank-5": 0.071,
            "Rank-10": 0.100,
            "Rank-20": 0.145,
            "mAP": np.nan,
            "Vis": 4.75,
        },
        {
            "dataset": "Mean",
            "test": "yolo26l-v10-240226.pt",
            "grid": "mean",
            "Num id": 526,
            "Num Img": 7930,
            "Missed": 25.2,
            "Emb": 7900,
            "Q total": 526,
            "Q valid": 526,
            "Rank-1": 0.100,
            "Rank-5": 0.168,
            "Rank-10": 0.218,
            "Rank-20": 0.272,
            "mAP": 0.020,
            "Vis": 4.75,
        },
    ]

    reid_by_model = {
        item["test"][:-3] if item["test"].endswith(".pt") else item["test"]: item
        for item in reid_eval_rows
    }
    alias_lookup = {
        "yolo11l (stock)": "yolo11l",
        "yolo26l (stock)": "yolo26l",
        "yolo11s (stock)": "yolo11s",
        "yolo26s (stock)": "yolo26s",
        "yolo26n (stock)": "yolo26n",
    }

    for row in rows:
        lookup_key = alias_lookup.get(row["Model"], row["Model"])
        reid = reid_by_model.get(lookup_key)
        if reid is None:
            row["Rank-1"] = np.nan
            row["Rank-5"] = np.nan
            row["Rank-10"] = np.nan
            row["Rank-20"] = np.nan
            row["ReID mAP"] = np.nan
            row["ReID Dataset"] = ""
            row["ReID Test"] = ""
            row["ReID Grid"] = ""
            row["ReID Num id"] = np.nan
            row["ReID Num Img"] = np.nan
            row["ReID Missed"] = np.nan
            row["ReID Emb"] = np.nan
            row["ReID Q total"] = np.nan
            row["ReID Q valid"] = np.nan
            row["ReID Vis"] = np.nan
            continue

        row["Rank-1"] = reid["Rank-1"]
        row["Rank-5"] = reid["Rank-5"]
        row["Rank-10"] = reid["Rank-10"]
        row["Rank-20"] = reid["Rank-20"]
        row["ReID mAP"] = reid["mAP"]
        row["ReID Dataset"] = reid["dataset"]
        row["ReID Test"] = reid["test"]
        row["ReID Grid"] = reid["grid"]
        row["ReID Num id"] = reid["Num id"]
        row["ReID Num Img"] = reid["Num Img"]
        row["ReID Missed"] = reid["Missed"]
        row["ReID Emb"] = reid["Emb"]
        row["ReID Q total"] = reid["Q total"]
        row["ReID Q valid"] = reid["Q valid"]
        row["ReID Vis"] = reid["Vis"]

    return pd.DataFrame(rows)


def _format_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    float_cols = [c for c in display_df.columns if c not in ("Model", "End2End")]
    for col in float_cols:
        if col in ("Params (M)", "GFLOPs"):
            display_df[col] = display_df[col].map(
                lambda x: "-" if pd.isna(x) else f"{x:.2f}".rstrip("0").rstrip(".")
            )
        else:
            display_df[col] = display_df[col].map(
                lambda x: "-" if pd.isna(x) else f"{x:.3f}"
            )
    return display_df


def _datatable_style_rgb(value: float, minval: float, maxval: float) -> tuple[int, int, int]:
    """Port of stuff/datatable.py colorize() logic (red-low / green-high)."""
    if pd.isna(value):
        return (245, 245, 245)

    thr = 0.02
    minval = min(minval, maxval - 2 * thr)
    delta_high = maxval - value
    delta_low = value - minval

    r, g, b = 255, 255, 255
    if delta_high < thr:
        v = int((delta_high * 255) / thr)
        r = v
        b = v
    elif delta_low < thr:
        v = int((delta_low * 255) / thr)
        g = v
        b = v
    return (r, g, b)


def _cell_color_for_metric(series: pd.Series, value: float) -> tuple[float, float, float, float]:
    if pd.isna(value):
        return (0.95, 0.95, 0.95, 1.0)

    valid = series.dropna().to_numpy(dtype=float)
    if len(valid) == 0:
        return (1.0, 1.0, 1.0, 1.0)

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    if vmax <= vmin:
        return (1.0, 1.0, 1.0, 1.0)

    r, g, b = _datatable_style_rgb(float(value), vmin, vmax)
    base = np.array([r, g, b], dtype=float) / 255.0
    # Soften terminal-style colors for table cell backgrounds.
    bg = 0.40 + 0.60 * base
    return (float(bg[0]), float(bg[1]), float(bg[2]), 1.0)


def render() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "images" / "results_weights_table.png"

    df = _build_dataframe()
    display_columns = [
        "Model",
        "End2End",
        "Params (M)",
        "GFLOPs",
        "Person",
        "Face",
        "Vehicle",
        "PoseL",
        "FaceKPL",
        "AttrMain",
        "ClrTop",
        "ClrBot",
        "Weapon",
        "Threat",
        "FIQA",
        "Rank-1",
        "ReID mAP",
    ]
    plot_df = df[display_columns].copy()
    display_df = _format_display(plot_df)

    fig_w = 22
    fig_h = 0.54 * (len(display_df) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_position([0.01, 0.01, 0.98, 0.98])

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.40)

    nrows, ncols = display_df.shape
    metric_cols = [c for c in plot_df.columns if c not in ("Model", "End2End")]
    col_widths = {
        "Model": 0.12,
        "End2End": 0.06,
        "Params (M)": 0.06,
        "GFLOPs": 0.06,
        "Rank-1": 0.055,
        "ReID mAP": 0.060,
    }
    default_metric_width = 0.050

    for col_idx, col_name in enumerate(display_df.columns):
        cell = table[0, col_idx]
        cell.set_facecolor("#1f2937")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#111827")
        cell.set_linewidth(1.0)

        width = col_widths.get(col_name, default_metric_width)
        for row_idx in range(0, nrows + 1):
            table[row_idx, col_idx].set_width(width)
        if col_name == "Model":
            cell.set_text_props(color="white", weight="bold", ha="left")

    for row in range(1, nrows + 1):
        zebra = "#f8fafc" if row % 2 == 0 else "white"
        for col in range(ncols):
            col_name = display_df.columns[col]
            cell = table[row, col]
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.6)

            if col_name == "Model":
                cell.set_facecolor(zebra)
                cell.set_text_props(ha="left", va="center")
            elif col_name == "End2End":
                is_e2e = df.iloc[row - 1][col_name] == "Yes"
                cell.set_facecolor("#d6f5d6" if is_e2e else "#ffe0e0")
                cell.set_text_props(weight="bold")
            elif col_name in metric_cols:
                value = (
                    float(plot_df.iloc[row - 1][col_name])
                    if pd.notna(plot_df.iloc[row - 1][col_name])
                    else np.nan
                )
                cell.set_facecolor(_cell_color_for_metric(plot_df[col_name], value))
            else:
                cell.set_facecolor(zebra)

    ax.set_title(
        "YOLO-DPAR Results and Weights (geometric mean over 11 validation sets)",
        fontsize=13,
        fontweight="bold",
        pad=6,
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    output = render()
    print(f"Wrote: {output}")
