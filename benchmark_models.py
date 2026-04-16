#!/usr/bin/env python3
"""
Benchmark YOLO-DPAR checkpoints with Ultralytics benchmark() in a temp workspace.

This wrapper is designed for Ubon's custom pose+attribute+reid models and provides:
- Runtime compatibility patches for `task=posereid`.
- Per-model temporary copy so export artifacts do not pollute source model folders.
- Programmatic aggregation of benchmark output into one combined table (CSV + JSON).
- Optional tiny smoke dataset generation for fast validation of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark


@contextmanager
def pushd(path: Path):
    """Temporarily change working directory."""
    old = Path.cwd()
    try:
        path.mkdir(parents=True, exist_ok=True)
        # os.chdir accepts str paths; Path is acceptable in modern python
        import os

        os.chdir(path)
        yield
    finally:
        import os

        os.chdir(old)


def patch_posereid_benchmark_compat() -> None:
    """
    Patch Ultralytics mappings and pose validator for PoseReID benchmark support.

    Why:
    - Upstream TASK2METRIC/TASK2DATA mappings may not include task='posereid'.
    - PoseValidator assumes extra channels are only [attrs + keypoints (+row_idx)].
      PoseReID adds reid embedding channels, which can break keypoint reshape.
    """
    from ultralytics.cfg import TASK2DATA, TASK2METRIC
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.models.yolo.pose.val import PoseValidator

    TASK2METRIC.setdefault("posereid", TASK2METRIC.get("pose", "metrics/mAP50-95(P)"))
    TASK2DATA.setdefault("posereid", TASK2DATA.get("pose", "coco8-pose.yaml"))

    if getattr(PoseValidator, "_ubon_posereid_patch", False):
        return

    def _patched_pose_postprocess(self: PoseValidator, preds):  # noqa: ANN001
        preds = DetectionValidator.postprocess(self, preds)
        nk = int(self.kpt_shape[0] * self.kpt_shape[1])
        for pred in preds:
            extra = pred.pop("extra")
            attr_nc = int(getattr(self, "attr_nc", 0) or 0)
            if attr_nc == 0:
                # Fallback for non-PyTorch backends where head metadata may be unavailable.
                data_attr_nc = int(getattr(self, "data", {}).get("attr_nc", 0) or 0)
                if data_attr_nc > 0 and extra.shape[1] >= data_attr_nc + nk:
                    attr_nc = data_attr_nc

            if attr_nc:
                if extra.shape[1] < attr_nc:
                    raise RuntimeError(
                        f"Invalid attr slice: extra has {extra.shape[1]} channels, expected at least {attr_nc}."
                    )
                pred["attr"] = extra[:, :attr_nc]
                extra = extra[:, attr_nc:]

            # Keep exactly nk keypoint channels and ignore any trailing channels
            # (e.g. PoseReID embeddings and/or e2e row index).
            if extra.shape[1] < nk:
                raise RuntimeError(
                    f"Invalid keypoint slice: extra has {extra.shape[1]} channels after attrs, expected at least {nk}."
                )
            pred["keypoints"] = extra[:, :nk].view(-1, *self.kpt_shape)
        return preds

    PoseValidator.postprocess = _patched_pose_postprocess
    PoseValidator._ubon_posereid_patch = True


def resolve_dataset_paths(dataset_yaml: Path) -> tuple[dict[str, Any], Path, Path]:
    """Resolve absolute image/label roots from a dataset yaml."""
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", dataset_yaml.parent))
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()
    val_rel = Path(cfg["val"])
    val_images = (base / val_rel).resolve()
    val_labels = (base / val_rel.parent / "labels").resolve()
    return cfg, val_images, val_labels


def build_smoke_dataset(source_yaml: Path, sample_count: int, out_root: Path) -> Path:
    """
    Build a tiny validation-only dataset clone for fast smoke testing.

    Copies first N val images and matching labels to a temporary dataset structure.
    """
    cfg, val_images, val_labels = resolve_dataset_paths(source_yaml)
    out_ds = out_root / "smoke_dataset"
    out_val_images = out_ds / "val" / "images"
    out_val_labels = out_ds / "val" / "labels"
    out_val_images.mkdir(parents=True, exist_ok=True)
    out_val_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp") for p in val_images.glob(ext)])
    if not image_files:
        raise FileNotFoundError(f"No validation images found under {val_images}")

    # Prefer labeled samples so validation metrics are meaningful.
    labeled = []
    for img in image_files:
        lbl = val_labels / f"{img.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
            labeled.append((img, lbl))
        if len(labeled) >= sample_count:
            break

    if not labeled:
        raise RuntimeError(f"No labeled validation samples found in {val_labels}")

    for img, lbl in labeled:
        shutil.copy2(img, out_val_images / img.name)
        shutil.copy2(lbl, out_val_labels / lbl.name)

    smoke_cfg = dict(cfg)
    smoke_cfg["path"] = str(out_ds)
    smoke_cfg["train"] = "val/images"
    smoke_cfg["val"] = "val/images"
    smoke_yaml = out_root / "smoke_dataset.yaml"
    smoke_yaml.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False), encoding="utf-8")
    return smoke_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark yolo-dpar models with Ultralytics benchmark().")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory containing .pt files.")
    parser.add_argument("--model-glob", type=str, default="*.pt", help="Glob pattern for model selection.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Benchmark image size.")
    parser.add_argument("--device", type=str, default="0", help="Benchmark device, e.g. 0, cpu, 0,1.")
    parser.add_argument(
        "--formats",
        type=str,
        default="-,onnx,engine",
        help="Comma-separated benchmark formats, e.g. '-,onnx,engine'.",
    )
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="Enable FP16.")
    parser.add_argument(
        "--onnx-simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ONNX graph simplification when format=onnx.",
    )
    parser.add_argument(
        "--engine-workspace",
        type=float,
        default=8.0,
        help="TensorRT workspace size in GB when format=engine.",
    )
    parser.add_argument("--limit-models", type=int, default=0, help="If >0, only benchmark first N models.")
    parser.add_argument(
        "--smoke-val-images",
        type=int,
        default=0,
        help="If >0, build tiny val set with N images for fast smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks"),
        help="Directory for combined outputs.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep per-model temporary benchmark directories.")
    return parser.parse_args()


def get_person_class_index(dataset_yaml: Path) -> int | None:
    """Resolve class index for 'person' in dataset yaml names."""
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if isinstance(names, list):
        for i, name in enumerate(names):
            if str(name).strip().lower() == "person":
                return i
    elif isinstance(names, dict):
        for k, name in names.items():
            if str(name).strip().lower() == "person":
                try:
                    return int(k)
                except Exception:
                    continue
    return None


def _class_ap50(metric_obj: Any, class_idx: int | None) -> float | None:
    """Extract AP50 for one class from a metric object using ap_class_index + all_ap."""
    if class_idx is None:
        return None
    ap_classes = getattr(metric_obj, "ap_class_index", None)
    all_ap = getattr(metric_obj, "all_ap", None)
    if ap_classes is None or all_ap is None:
        return None
    try:
        for row_idx, cls in enumerate(ap_classes):
            if int(cls) == int(class_idx):
                return float(all_ap[row_idx, 0])  # AP50 is IoU index 0
    except Exception:
        return None
    return None


def artifact_path_for_format(copied_model: Path, fmt: str) -> Path:
    """Return expected artifact path for a benchmark format request."""
    if fmt == "-":
        return copied_model
    if fmt == "onnx":
        return copied_model.with_suffix(".onnx")
    if fmt == "engine":
        return copied_model.with_suffix(".engine")
    # Fallback: try checkpoint itself if unknown format (may still be loadable).
    return copied_model


def collect_posereid_extra_metrics(
    model_artifact: Path,
    data_yaml: Path,
    imgsz: int,
    half: bool,
    device: str,
    person_class_index: int | None,
    source_task: str | None = None,
) -> dict[str, Any]:
    """
    Collect extra PoseReID benchmark metrics requested by the user.

    Metrics:
    - metrics/mAP50_person(B): non-pose AP50 for person class only
    - metrics/mAP50_person(P): pose AP50 for person class only
    - metrics/mAP50_all(B): non-pose AP50 for all classes
    """
    model = YOLO(str(model_artifact), task=source_task) if source_task else YOLO(str(model_artifact))
    task_name = source_task or model.task
    if task_name != "posereid":
        return {}

    results = model.val(
        data=str(data_yaml),
        batch=1,
        imgsz=imgsz,
        plots=False,
        device=device,
        half=half,
        verbose=False,
        conf=0.001,
    )
    return {
        "metrics/mAP50_person(B)": _class_ap50(results.box, person_class_index),
        "metrics/mAP50_person(P)": _class_ap50(results.pose, person_class_index),
        "metrics/mAP50_all(B)": float(results.results_dict.get("metrics/mAP50(B)"))
        if results.results_dict.get("metrics/mAP50(B)") is not None
        else None,
    }


def export_kwargs_for_format(fmt: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if fmt == "onnx":
        kwargs["simplify"] = args.onnx_simplify
    if fmt == "engine":
        kwargs["workspace"] = args.engine_workspace
    return kwargs


def _to_float(value: Any) -> float | None:
    """Best-effort conversion to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if not v or v == "-":
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _datatable_style_rgb(value: float, minval: float, maxval: float) -> tuple[int, int, int]:
    """Port of repo table coloring (red-low / green-high)."""
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


def _cell_color(value: float | None, values: list[float], higher_better: bool = True) -> tuple[float, float, float, float]:
    """Color a numeric cell using the same style as existing table scripts."""
    if value is None or not values:
        return (0.95, 0.95, 0.95, 1.0)

    # Convert lower-is-better metrics into a higher-is-better score space.
    if not higher_better:
        score_values = [-x for x in values]
        score = -value
    else:
        score_values = values
        score = value

    vmin = min(score_values)
    vmax = max(score_values)
    if vmax <= vmin:
        return (1.0, 1.0, 1.0, 1.0)

    r, g, b = _datatable_style_rgb(score, vmin, vmax)
    base = [r / 255.0, g / 255.0, b / 255.0]
    # Soften terminal-like colors for readable backgrounds.
    bg = [0.40 + 0.60 * x for x in base]
    return (float(bg[0]), float(bg[1]), float(bg[2]), 1.0)


def render_results_table_image(records: list[dict[str, Any]], output_path: Path) -> Path | None:
    """
    Render a styled PNG summary table for benchmark results.

    Output is intentionally concise and presentation-focused, while CSV/JSON keep full raw payloads.
    """
    if not records:
        return None

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[warn] skipping table image render (missing deps): {exc}")
        return None

    df = pd.DataFrame(records)
    if df.empty:
        return None

    metric_cols_all = [c for c in df.columns if c.startswith("metrics/")]
    preferred_metric_order = [
        "metrics/mAP50_person(B)",
        "metrics/mAP50_person(P)",
        "metrics/mAP50_all(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(P)",
        "metrics/mAP50(A)",
    ]
    metric_cols = [c for c in preferred_metric_order if c in metric_cols_all] + [
        c for c in sorted(metric_cols_all) if c not in preferred_metric_order
    ]

    base_cols = [
        col
        for col in ("model", "format", "format_request", "status", "size_mb", "inference_ms_per_im", "fps")
        if col in df.columns
    ]
    table_cols = base_cols + metric_cols
    tdf = df[table_cols].copy()
    if "status" in tdf.columns:
        tdf["status"] = tdf["status"].map(
            lambda s: "OK" if "✅" in str(s) else ("WARN" if str(s) in {"❎", "❌"} else str(s))
        )

    # Compact column labels for the image table.
    label_map = {
        "model": "Model",
        "format": "Format",
        "format_request": "ReqFmt",
        "status": "Status",
        "size_mb": "Size MB",
        "inference_ms_per_im": "Infer ms/im",
        "fps": "FPS",
        "metrics/mAP50_person(B)": "mAP50 Person Box",
        "metrics/mAP50_person(P)": "mAP50 Person Pose",
        "metrics/mAP50_all(B)": "mAP50 All Box",
        "metrics/mAP50(B)": "mAP50 Box",
        "metrics/mAP50(P)": "mAP50 Pose",
        "metrics/mAP50-95(B)": "mAP50-95 Box",
        "metrics/mAP50-95(P)": "mAP50-95 Pose",
        "metrics/mAP50(A)": "mAP50 Attr",
    }
    tdf.rename(columns={k: v for k, v in label_map.items() if k in tdf.columns}, inplace=True)

    numeric_cols = [
        c for c in tdf.columns if c not in {"Model", "Format", "ReqFmt", "Status"}
    ]
    numeric_values: dict[str, list[float]] = {}
    for col in numeric_cols:
        vals = []
        for raw in tdf[col].tolist():
            f = _to_float(raw)
            if f is not None:
                vals.append(f)
        numeric_values[col] = vals

    # Display formatting
    display_df = tdf.copy()
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: "-" if _to_float(x) is None else f"{float(_to_float(x)):.4f}".rstrip("0").rstrip("."))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(13.0, min(30.0, 4.0 + 1.35 * len(display_df.columns)))
    fig_h = max(3.0, 0.56 * (len(display_df) + 2))
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
    table.scale(1, 1.38)

    nrows, ncols = display_df.shape
    col_widths = {
        "Model": 0.20,
        "Format": 0.07,
        "ReqFmt": 0.06,
        "Status": 0.06,
        "Size MB": 0.07,
        "Infer ms/im": 0.09,
        "FPS": 0.06,
    }
    default_metric_width = 0.09

    for col_idx, col_name in enumerate(display_df.columns):
        head = table[0, col_idx]
        head.set_facecolor("#1f2937")
        head.set_text_props(color="white", weight="bold")
        head.set_edgecolor("#111827")
        head.set_linewidth(1.0)
        if col_name == "Model":
            head.set_text_props(color="white", weight="bold", ha="left")

        width = col_widths.get(col_name, default_metric_width)
        for row_idx in range(0, nrows + 1):
            table[row_idx, col_idx].set_width(width)

    for row_idx in range(1, nrows + 1):
        zebra = "#f8fafc" if row_idx % 2 == 0 else "white"
        for col_idx in range(ncols):
            col_name = display_df.columns[col_idx]
            cell = table[row_idx, col_idx]
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.6)

            if col_name == "Model":
                cell.set_facecolor(zebra)
                cell.set_text_props(ha="left")
                continue
            if col_name == "Status":
                status = str(tdf.iloc[row_idx - 1][col_name])
                if status == "OK":
                    cell.set_facecolor("#d6f5d6")
                elif status in {"WARN", "CRASH"}:
                    cell.set_facecolor("#ffe0e0")
                else:
                    cell.set_facecolor("#fff8dc")
                cell.set_text_props(weight="bold")
                continue

            if col_name in numeric_cols:
                raw = tdf.iloc[row_idx - 1][col_name]
                value = _to_float(raw)
                higher_better = col_name not in {"Size MB", "Infer ms/im"}
                cell.set_facecolor(_cell_color(value, numeric_values.get(col_name, []), higher_better=higher_better))
            else:
                cell.set_facecolor(zebra)

    ax.set_title("YOLO-DPAR Benchmark Results", fontsize=13, fontweight="bold", pad=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    patch_posereid_benchmark_compat()

    models_dir = args.models_dir.resolve()
    data_yaml = args.data.resolve()
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    models = sorted(models_dir.glob(args.model_glob))
    if args.limit_models and args.limit_models > 0:
        models = models[: args.limit_models]
    if not models:
        raise FileNotFoundError(f"No models matched {args.model_glob} in {models_dir}")

    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.resolve() / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_for_run = data_yaml
    if args.smoke_val_images and args.smoke_val_images > 0:
        smoke_root = run_dir / "smoke_data"
        smoke_root.mkdir(parents=True, exist_ok=True)
        data_for_run = build_smoke_dataset(data_yaml, args.smoke_val_images, smoke_root)
        print(f"[info] smoke dataset created: {data_for_run}")
    person_class_index = get_person_class_index(data_for_run)

    records: list[dict[str, Any]] = []
    for model_path in models:
        try:
            source_model = YOLO(str(model_path))
            source_task = source_model.task
        except Exception:
            source_task = None
        for fmt in formats:
            tmp_dir = Path(tempfile.mkdtemp(prefix="ubon_bench_model_"))
            copied_model = tmp_dir / model_path.name
            shutil.copy2(model_path, copied_model)
            print(f"[run] model={model_path.name} format={fmt} temp={tmp_dir}")

            try:
                with pushd(tmp_dir):
                    df = benchmark(
                        model=str(copied_model),
                        data=str(data_for_run),
                        imgsz=args.imgsz,
                        half=args.half,
                        device=args.device,
                        format=fmt,
                        **export_kwargs_for_format(fmt, args),
                    )
                metric_cols = [c for c in df.columns if c.startswith("metrics/")]
                for row in df.to_dicts():
                    benchmark_metric_payload = {col: row.get(col, None) for col in metric_cols}
                    extra_metrics: dict[str, Any] = {}
                    artifact_path = artifact_path_for_format(copied_model, fmt)
                    if artifact_path.exists():
                        try:
                            extra_metrics = collect_posereid_extra_metrics(
                                model_artifact=artifact_path,
                                data_yaml=data_for_run,
                                imgsz=args.imgsz,
                                half=args.half,
                                device=args.device,
                                person_class_index=person_class_index,
                                source_task=source_task,
                            )
                        except Exception as metric_exc:
                            extra_metrics = {"pose_metric_error": str(metric_exc)}

                    records.append(
                        {
                            "model": model_path.name,
                            "model_path": str(model_path),
                            "format_request": fmt,
                            "format": row.get("Format"),
                            "status": row.get("Status❔"),
                            "size_mb": row.get("Size (MB)"),
                            "inference_ms_per_im": row.get("Inference time (ms/im)"),
                            "fps": row.get("FPS"),
                            "temp_dir": str(tmp_dir),
                            **benchmark_metric_payload,
                            **extra_metrics,
                        }
                    )
            except Exception as exc:  # keep iterating on failures
                records.append(
                    {
                        "model": model_path.name,
                        "model_path": str(model_path),
                        "format_request": fmt,
                        "format": fmt,
                        "status": "CRASH",
                        "size_mb": None,
                        "metric_name": "",
                        "metric_value": None,
                        "inference_ms_per_im": None,
                        "fps": None,
                        "temp_dir": str(tmp_dir),
                        "error": str(exc),
                    }
                )
                print(f"[error] model={model_path.name} format={fmt}: {exc}")
            finally:
                if not args.keep_temp:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Write combined outputs without requiring pandas/polars.
    json_path = run_dir / "combined_results.json"
    csv_path = run_dir / "combined_results.csv"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    import csv

    fieldnames = sorted({k for r in records for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    image_path = run_dir / "combined_results_table.png"
    rendered_image = render_results_table_image(records, image_path)

    print(f"[done] results rows={len(records)}")
    print(f"[done] csv={csv_path}")
    print(f"[done] json={json_path}")
    if rendered_image is not None:
        print(f"[done] table_image={rendered_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
