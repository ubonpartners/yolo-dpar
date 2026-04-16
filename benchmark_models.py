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
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

STOCK_DETECT_MODELS = ("yolo26l.pt", "yolo26s.pt")
STOCK_POSE_MODELS = ("yolo26l-pose.pt", "yolo26s-pose.pt")
DEFAULT_STOCK_MODELS = (*STOCK_DETECT_MODELS, *STOCK_POSE_MODELS)
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer_file(path: Path) -> bool:
    """Return True when a file is a Git LFS pointer text blob."""
    try:
        if not path.is_file():
            return False
        with path.open("rb") as f:
            head = f.read(256)
    except OSError:
        return False
    return head.startswith(LFS_POINTER_PREFIX)


def raise_if_lfs_pointer_files(paths: list[Path], include_glob: str = "models/*.pt") -> None:
    """Fail fast with an actionable message when checkpoints are LFS pointers."""
    pointer_paths = [p for p in paths if is_git_lfs_pointer_file(p)]
    if not pointer_paths:
        return

    listed = "\n".join(f"  - {p}" for p in pointer_paths)
    raise RuntimeError(
        "Detected Git LFS pointer file(s) instead of real .pt checkpoints:\n"
        f"{listed}\n\n"
        "These files are text pointers (typically ~130 bytes), so PyTorch cannot load them.\n"
        "Install/fetch Git LFS weights, then re-run:\n"
        "  git lfs install\n"
        f"  git lfs pull --include=\"{include_glob}\"\n"
        "If git-lfs is missing on Ubuntu/Debian:\n"
        "  sudo apt-get update && sudo apt-get install -y git-lfs"
    )


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


def ensure_train_key_for_benchmark(dataset_yaml: Path, out_root: Path) -> Path:
    """
    Ensure dataset yaml has a train key for Ultralytics benchmark() checks.

    Some val-only datasets intentionally omit `train`. Ultralytics benchmark()
    currently enforces both keys, so we create a runtime copy with train=val.
    """
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    if cfg.get("train"):
        return dataset_yaml
    if not cfg.get("val"):
        raise KeyError(f"{dataset_yaml} is missing required 'val' key.")

    patched = dict(cfg)
    patched["train"] = cfg["val"]
    patched_root = out_root / "runtime_data"
    patched_root.mkdir(parents=True, exist_ok=True)
    patched_yaml = patched_root / f"{dataset_yaml.stem}_benchmark.yaml"
    patched_yaml.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
    print(f"[info] dataset yaml missing train; wrote runtime benchmark yaml: {patched_yaml}")
    return patched_yaml


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
    parser.add_argument(
        "--include-stock-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include stock Ultralytics checkpoints "
            "(yolo26l.pt, yolo26l-pose.pt, yolo26s.pt, yolo26s-pose.pt), auto-downloaded if missing."
        ),
    )
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
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run benchmark jobs in parallel across all visible GPUs (default: on).",
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


def _iter_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _parse_source_kpt_shape(cfg: dict[str, Any]) -> tuple[int, int] | None:
    kpt_shape = cfg.get("kpt_shape")
    if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) == 2:
        try:
            return int(kpt_shape[0]), int(kpt_shape[1])
        except Exception:
            return None
    return None


def _source_attr_len(cfg: dict[str, Any]) -> int:
    """Return per-label attribute column count used before keypoints in split format."""
    attributes = bool(cfg.get("attributes", False))
    attr_label_format = str(cfg.get("attr_label_format", "combined")).lower()
    attr_nc = int(cfg.get("attr_nc", 0) or 0)
    return attr_nc if attributes and attr_label_format == "split" else 0


def _extract_person_pose_flat(
    parts: list[str],
    source_kpt_shape: tuple[int, int] | None,
    dst_kpt_shape: tuple[int, int],
    source_attr_len: int = 0,
) -> list[float] | None:
    if source_kpt_shape is None:
        return None

    src_nk, src_nd = source_kpt_shape
    dst_nk, dst_nd = dst_kpt_shape
    kpt_start = 5 + int(source_attr_len or 0)
    need = kpt_start + src_nk * src_nd
    if len(parts) < need:
        return None

    try:
        raw = [float(x) for x in parts[kpt_start : kpt_start + src_nk * src_nd]]
    except Exception:
        return None
    kpts = [raw[i * src_nd : (i + 1) * src_nd] for i in range(src_nk)]

    # v10 merged labels are [5 face kpts + 17 person pose kpts].
    if src_nk == 22 and dst_nk == 17:
        chosen = kpts[5:22]
    elif src_nk >= dst_nk:
        chosen = kpts[src_nk - dst_nk :]
    else:
        return None

    out: list[float] = []
    for kp in chosen:
        vals = kp[: min(src_nd, dst_nd)]
        if len(vals) < dst_nd:
            # Fill missing visibility channel when needed.
            vals = vals + [1.0] * (dst_nd - len(vals))
        out.extend(vals[:dst_nd])
    return out


def build_stock_intersection_datasets(source_yaml: Path, out_root: Path, pose_kpt_shape: tuple[int, int] = (17, 3)) -> dict[str, Path]:
    """
    Build stock-model-compatible dataset views from merged labels.

    - detect view: person-only boxes (for stock detect models)
    - pose view: person-only boxes + 17-keypoint pose (for stock pose models)
    """
    cfg, val_images, val_labels = resolve_dataset_paths(source_yaml)
    person_idx = get_person_class_index(source_yaml)
    if person_idx is None:
        raise ValueError(f"Could not resolve 'person' class index from {source_yaml}")

    source_kpt_shape = _parse_source_kpt_shape(cfg)
    source_attr_len = _source_attr_len(cfg)
    images = _iter_images(val_images)
    if not images:
        raise FileNotFoundError(f"No validation images found under {val_images}")

    base = out_root / "stock_intersection"
    detect_root = base / "detect"
    pose_root = base / "pose"
    detect_img_root = detect_root / "val" / "images"
    detect_lbl_root = detect_root / "val" / "labels"
    pose_img_root = pose_root / "val" / "images"
    pose_lbl_root = pose_root / "val" / "labels"
    detect_img_root.mkdir(parents=True, exist_ok=True)
    detect_lbl_root.mkdir(parents=True, exist_ok=True)
    pose_img_root.mkdir(parents=True, exist_ok=True)
    pose_lbl_root.mkdir(parents=True, exist_ok=True)

    for img in images:
        rel = img.relative_to(val_images)
        src_lbl = (val_labels / rel).with_suffix(".txt")
        det_img = detect_img_root / rel
        pose_img = pose_img_root / rel
        det_lbl = (detect_lbl_root / rel).with_suffix(".txt")
        pose_lbl = (pose_lbl_root / rel).with_suffix(".txt")
        det_lbl.parent.mkdir(parents=True, exist_ok=True)
        pose_lbl.parent.mkdir(parents=True, exist_ok=True)

        _symlink_or_copy(img, det_img)
        _symlink_or_copy(img, pose_img)

        det_lines: list[str] = []
        pose_lines: list[str] = []
        if src_lbl.exists():
            for line in src_lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = [x for x in line.strip().split() if x]
                if len(parts) < 5:
                    continue
                try:
                    cls_idx = int(float(parts[0]))
                except Exception:
                    continue
                if cls_idx != person_idx:
                    continue

                det_lines.append(" ".join(["0", *parts[1:5]]))

                pose_flat = _extract_person_pose_flat(parts, source_kpt_shape, pose_kpt_shape, source_attr_len)
                if pose_flat is not None:
                    pose_lines.append(" ".join(["0", *parts[1:5], *(f"{v:.6g}" for v in pose_flat)]))

        det_lbl.write_text("\n".join(det_lines) + ("\n" if det_lines else ""), encoding="utf-8")
        pose_lbl.write_text("\n".join(pose_lines) + ("\n" if pose_lines else ""), encoding="utf-8")

    detect_yaml = detect_root / "dataset.yaml"
    pose_yaml = pose_root / "dataset.yaml"

    detect_cfg = {
        "path": str(detect_root),
        "train": "val/images",
        "val": "val/images",
        "nc": 1,
        "names": ["person"],
    }
    pose_cfg = {
        "path": str(pose_root),
        "train": "val/images",
        "val": "val/images",
        "nc": 1,
        "names": ["person"],
        "kpt_shape": list(pose_kpt_shape),
    }
    detect_yaml.write_text(yaml.safe_dump(detect_cfg, sort_keys=False), encoding="utf-8")
    pose_yaml.write_text(yaml.safe_dump(pose_cfg, sort_keys=False), encoding="utf-8")

    return {"detect": detect_yaml, "pose": pose_yaml}


def resolve_model_entries(models_dir: Path, model_glob: str, include_stock_models: bool) -> list[dict[str, str]]:
    """
    Resolve benchmark model sources into local paths ready for temp-copy benchmarking.

    Returns list entries:
      - model: display name (filename)
      - source: original spec (path or model name)
      - resolved_path: absolute local checkpoint path
    """
    entries: list[dict[str, str]] = []
    local_paths = sorted(models_dir.glob(model_glob))
    raise_if_lfs_pointer_files(local_paths, include_glob=f"{models_dir.name}/{model_glob}")
    for p in local_paths:
        entries.append(
            {
                "model": p.name,
                "source": str(p),
                "resolved_path": str(p.resolve()),
            }
        )

    existing_names = {e["model"] for e in entries}
    if include_stock_models:
        for spec in DEFAULT_STOCK_MODELS:
            model_name = Path(spec).name
            if model_name in existing_names:
                continue
            print(f"[info] ensuring stock model is available: {spec}")
            stock_model = YOLO(spec)  # triggers auto-download if missing
            resolved = stock_model.pt_path or stock_model.ckpt_path or spec
            resolved_path = Path(str(resolved)).expanduser().resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"Stock model resolved path does not exist: {resolved_path}")
            raise_if_lfs_pointer_files([resolved_path], include_glob=f"{models_dir.name}/{model_glob}")
            entries.append(
                {
                    "model": model_name,
                    "source": spec,
                    "resolved_path": str(resolved_path),
                }
            )
            existing_names.add(model_name)
    return entries


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


def collect_intersection_extra_metrics(
    model_artifact: Path,
    data_yaml: Path,
    imgsz: int,
    half: bool,
    device: str,
    person_class_index: int | None,
    source_task: str | None = None,
) -> dict[str, Any]:
    """
    Collect extra intersection metrics for detect/pose/posereid tasks.
    """
    model = YOLO(str(model_artifact), task=source_task) if source_task else YOLO(str(model_artifact))
    task_name = source_task or model.task
    if task_name not in {"detect", "pose", "posereid"}:
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
    out: dict[str, Any] = {}
    if results.results_dict.get("metrics/mAP50-95(B)") is not None:
        out["metrics/mAP50-95(B)"] = float(results.results_dict.get("metrics/mAP50-95(B)"))
    if results.results_dict.get("metrics/mAP50-95(P)") is not None:
        out["metrics/mAP50-95(P)"] = float(results.results_dict.get("metrics/mAP50-95(P)"))
    if results.results_dict.get("metrics/mAP50(B)") is not None:
        out["metrics/mAP50_all(B)"] = float(results.results_dict.get("metrics/mAP50(B)"))
    if hasattr(results, "box"):
        out["metrics/mAP50_person(B)"] = _class_ap50(results.box, person_class_index)

    if task_name in {"pose", "posereid"}:
        if results.results_dict.get("metrics/mAP50(P)") is not None:
            out["metrics/mAP50_all(P)"] = float(results.results_dict.get("metrics/mAP50(P)"))
        if hasattr(results, "pose"):
            out["metrics/mAP50_person(P)"] = _class_ap50(results.pose, person_class_index)
    return out


def choose_data_for_model(
    model_name: str,
    default_data_yaml: Path,
    stock_intersection: dict[str, Path] | None,
) -> Path:
    """Select benchmark data yaml per model with stock-model intersection handling."""
    if stock_intersection:
        if model_name in STOCK_DETECT_MODELS:
            return stock_intersection["detect"]
        if model_name in STOCK_POSE_MODELS:
            return stock_intersection["pose"]
    return default_data_yaml


def export_kwargs_for_format(fmt: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if fmt == "onnx":
        kwargs["simplify"] = args.onnx_simplify
    if fmt == "engine":
        kwargs["workspace"] = args.engine_workspace
    return kwargs


def detect_visible_gpu_indices() -> list[str]:
    """
    Detect visible GPU indices as strings ("0", "1", ...).

    Uses torch first, then falls back to nvidia-smi when available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            n = int(torch.cuda.device_count())
            if n > 0:
                return [str(i) for i in range(n)]
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            return [str(i) for i in range(len(lines))]
    except Exception:
        pass

    return []


def configure_ultralytics_console(plain_progress: bool) -> None:
    """
    Configure Ultralytics console behavior.

    In multiprocess mode we prefer plain logging (no in-place progress redraws)
    to avoid interleaved ANSI output from concurrent workers.
    """
    if not plain_progress:
        return
    os.environ["YOLO_VERBOSE"] = "False"
    try:
        import ultralytics.utils as uutils

        uutils.VERBOSE = False
    except Exception:
        pass


def run_single_benchmark_job(job: dict[str, Any], args: argparse.Namespace, device: str) -> list[dict[str, Any]]:
    """Run one (model, format) benchmark job and return result records."""
    model_name = str(job["model"])
    model_source = str(job["model_source"])
    model_resolved_path = Path(str(job["model_path"]))
    model_data_yaml = Path(str(job["data_yaml"]))
    model_person_idx = job.get("person_class_index")
    source_task = job.get("source_task")
    fmt = str(job["format_request"])
    job_idx = int(job["job_idx"])

    records: list[dict[str, Any]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ubon_bench_model_"))
    copied_model = tmp_dir / model_name
    shutil.copy2(model_resolved_path, copied_model)
    print(f"[run] model={model_name} format={fmt} device={device} temp={tmp_dir}")

    try:
        with pushd(tmp_dir):
            df = benchmark(
                model=str(copied_model),
                data=str(model_data_yaml),
                imgsz=args.imgsz,
                half=args.half,
                device=device,
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
                    extra_metrics = collect_intersection_extra_metrics(
                        model_artifact=artifact_path,
                        data_yaml=model_data_yaml,
                        imgsz=args.imgsz,
                        half=args.half,
                        device=device,
                        person_class_index=model_person_idx,
                        source_task=source_task,
                    )
                except Exception as metric_exc:
                    extra_metrics = {"pose_metric_error": str(metric_exc)}

            records.append(
                {
                    "_job_idx": job_idx,
                    "model": model_name,
                    "model_source": model_source,
                    "model_path": str(model_resolved_path),
                    "data_yaml": str(model_data_yaml),
                    "format_request": fmt,
                    "format": row.get("Format"),
                    "status": row.get("Status❔"),
                    "size_mb": row.get("Size (MB)"),
                    "inference_ms_per_im": row.get("Inference time (ms/im)"),
                    "fps": row.get("FPS"),
                    "temp_dir": str(tmp_dir),
                    "device": device,
                    **benchmark_metric_payload,
                    **extra_metrics,
                }
            )
    except Exception as exc:  # keep iterating on failures
        records.append(
            {
                "_job_idx": job_idx,
                "model": model_name,
                "model_source": model_source,
                "model_path": str(model_resolved_path),
                "data_yaml": str(model_data_yaml),
                "format_request": fmt,
                "format": fmt,
                "status": "CRASH",
                "size_mb": None,
                "inference_ms_per_im": None,
                "fps": None,
                "temp_dir": str(tmp_dir),
                "device": device,
                "error": str(exc),
            }
        )
        print(f"[error] model={model_name} format={fmt} device={device}: {exc}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return records


def _parallel_worker(
    work_q: mp.Queue,
    out_q: mp.Queue,
    worker_device: str,
    args: argparse.Namespace,
) -> None:
    """Worker loop pinned to one GPU device."""
    configure_ultralytics_console(plain_progress=True)
    patch_posereid_benchmark_compat()
    while True:
        job = work_q.get()
        if job is None:
            break
        try:
            out_q.put(run_single_benchmark_job(job, args, device=worker_device))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Keep coordinator alive even if worker sees an unexpected failure.
            out_q.put(
                [
                    {
                        "_job_idx": int(job.get("job_idx", -1)),
                        "model": str(job.get("model", "unknown")),
                        "model_source": str(job.get("model_source", "unknown")),
                        "model_path": str(job.get("model_path", "")),
                        "data_yaml": str(job.get("data_yaml", "")),
                        "format_request": str(job.get("format_request", "")),
                        "format": str(job.get("format_request", "")),
                        "status": "CRASH",
                        "size_mb": None,
                        "inference_ms_per_im": None,
                        "fps": None,
                        "temp_dir": None,
                        "device": worker_device,
                        "error": f"worker_error: {exc}",
                    }
                ]
            )


def run_jobs_serial(jobs: list[dict[str, Any]], args: argparse.Namespace, device: str) -> list[dict[str, Any]]:
    """Execute all jobs sequentially on one device."""
    records: list[dict[str, Any]] = []
    total = len(jobs)
    if total:
        print(f"[progress] jobs total={total} completed=0 remaining={total}")
    for idx, job in enumerate(jobs, start=1):
        records.extend(run_single_benchmark_job(job, args, device=device))
        remaining = total - idx
        print(f"[progress] jobs total={total} completed={idx} remaining={remaining}")
    return records


def run_jobs_parallel(jobs: list[dict[str, Any]], args: argparse.Namespace, gpu_devices: list[str]) -> list[dict[str, Any]]:
    """Execute jobs in parallel, one process per GPU, pulling from shared queue."""
    if not jobs:
        return []
    if not gpu_devices:
        return run_jobs_serial(jobs, args, device=str(args.device))

    ctx = mp.get_context("spawn")
    work_q = ctx.Queue()
    out_q = ctx.Queue()

    for job in jobs:
        work_q.put(job)
    for _ in gpu_devices:
        work_q.put(None)

    procs = [
        ctx.Process(target=_parallel_worker, args=(work_q, out_q, gpu, args), name=f"bench-gpu-{gpu}")
        for gpu in gpu_devices
    ]
    for p in procs:
        p.start()

    records: list[dict[str, Any]] = []
    expected = len(jobs)
    received = 0
    print(f"[progress] jobs total={expected} completed=0 remaining={expected}")

    try:
        while received < expected:
            try:
                batch = out_q.get(timeout=1.0)
            except queue.Empty:
                dead = [p for p in procs if (not p.is_alive()) and (p.exitcode not in (0, None))]
                if dead:
                    codes = ", ".join(f"{p.name}:{p.exitcode}" for p in dead)
                    raise RuntimeError(f"Parallel benchmark worker failed ({codes}).")
                continue
            records.extend(batch)
            received += 1
            remaining = expected - received
            print(f"[progress] jobs total={expected} completed={received} remaining={remaining}")
    except KeyboardInterrupt:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)
        raise
    finally:
        for p in procs:
            p.join()

    return records


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


def _dataset_label(path_like: Any) -> str:
    """Build a compact dataset label for table display."""
    raw = str(path_like or "").strip()
    if not raw:
        return "-"
    try:
        p = Path(raw)
    except Exception:
        return raw

    parts = p.parts
    if "stock_intersection" in parts:
        i = parts.index("stock_intersection")
        tail = parts[i + 1 :]
        return "/".join(tail[-2:]) if len(tail) >= 2 else "stock_intersection"
    if "smoke_data" in parts:
        return "smoke_dataset"
    return p.name


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
    image_excluded_metrics = {"metrics/mAP50_all(P)", "metrics/mAP50-95(P)"}
    metric_cols_all = [c for c in metric_cols_all if c not in image_excluded_metrics]
    preferred_metric_order = [
        "metrics/mAP50_person(B)",
        "metrics/mAP50_person(P)",
        "metrics/mAP50_all(B)",
        "metrics/mAP50_all(P)",
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
        for col in ("model", "data_yaml", "format", "format_request", "status", "size_mb", "inference_ms_per_im", "fps")
        if col in df.columns
    ]
    table_cols = base_cols + metric_cols
    tdf = df[table_cols].copy()
    if "data_yaml" in tdf.columns:
        tdf["data_yaml"] = tdf["data_yaml"].map(_dataset_label)
    if "status" in tdf.columns:
        tdf["status"] = tdf["status"].map(
            lambda s: "OK" if "✅" in str(s) else ("WARN" if str(s) in {"❎", "❌"} else str(s))
        )

    # Compact column labels for the image table.
    label_map = {
        "model": "Model",
        "data_yaml": "Dataset",
        "format": "Format",
        "format_request": "ReqFmt",
        "status": "Status",
        "size_mb": "Size MB",
        "inference_ms_per_im": "Infer ms/im",
        "fps": "FPS",
        "metrics/mAP50_person(B)": "mAP50 Person Box",
        "metrics/mAP50_person(P)": "mAP50 Person Pose",
        "metrics/mAP50_all(B)": "mAP50 All Box",
        "metrics/mAP50_all(P)": "mAP50 All Pose",
        "metrics/mAP50(B)": "mAP50 Box",
        "metrics/mAP50(P)": "mAP50 Pose",
        "metrics/mAP50-95(B)": "mAP50-95 Box",
        "metrics/mAP50-95(P)": "mAP50-95 Pose",
        "metrics/mAP50(A)": "mAP50 Attr",
    }
    tdf.rename(columns={k: v for k, v in label_map.items() if k in tdf.columns}, inplace=True)

    numeric_cols = [
        c for c in tdf.columns if c not in {"Model", "Dataset", "Format", "ReqFmt", "Status"}
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
        "Dataset": 0.12,
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

    model_entries = resolve_model_entries(models_dir, args.model_glob, args.include_stock_models)
    if args.limit_models and args.limit_models > 0:
        model_entries = model_entries[: args.limit_models]
    if not model_entries:
        raise FileNotFoundError(
            f"No benchmark models resolved (glob={args.model_glob}, include_stock_models={args.include_stock_models})."
        )

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
    data_for_run = ensure_train_key_for_benchmark(data_for_run, run_dir)
    stock_intersection = None
    if any(e["model"] in DEFAULT_STOCK_MODELS for e in model_entries):
        stock_intersection = build_stock_intersection_datasets(data_for_run, run_dir)
        print(f"[info] stock detect intersection data: {stock_intersection['detect']}")
        print(f"[info] stock pose intersection data: {stock_intersection['pose']}")

    prepared_models: list[dict[str, Any]] = []
    for model_entry in model_entries:
        model_name = model_entry["model"]
        model_source = model_entry["source"]
        model_resolved_path = Path(model_entry["resolved_path"])
        try:
            source_model = YOLO(str(model_resolved_path))
            source_task = source_model.task
        except Exception:
            source_task = None
        model_data_yaml = choose_data_for_model(model_name, data_for_run, stock_intersection)
        model_person_idx = get_person_class_index(model_data_yaml)
        prepared_models.append(
            {
                "model": model_name,
                "model_source": model_source,
                "model_path": str(model_resolved_path),
                "data_yaml": str(model_data_yaml),
                "person_class_index": model_person_idx,
                "source_task": source_task,
            }
        )

    jobs: list[dict[str, Any]] = []
    job_idx = 0
    for pm in prepared_models:
        for fmt in formats:
            jobs.append({**pm, "format_request": fmt, "job_idx": job_idx})
            job_idx += 1

    gpu_devices = detect_visible_gpu_indices()
    use_parallel = bool(args.parallel and len(gpu_devices) > 1)
    if use_parallel:
        configure_ultralytics_console(plain_progress=True)
        print(f"[info] parallel mode enabled: {len(gpu_devices)} GPU workers ({', '.join(gpu_devices)})")
        records = run_jobs_parallel(jobs, args, gpu_devices=gpu_devices)
    else:
        if args.parallel and len(gpu_devices) <= 1:
            reason = "no visible GPUs" if len(gpu_devices) == 0 else "single visible GPU"
            print(f"[info] parallel mode requested but using serial mode ({reason})")
        records = run_jobs_serial(jobs, args, device=str(args.device))

    records.sort(key=lambda r: int(r.get("_job_idx", 10**9)))
    for row in records:
        row.pop("_job_idx", None)

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
