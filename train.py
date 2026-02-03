"""
Standalone local training entrypoint for YOLO-DP/DPA/DPAR models.

This is adapted from `azureml/run_train.py` but intentionally removes:
- AzureML submission / datastores / mounts
- MLflow experiment wiring

It keeps the original "one YAML config drives everything" pattern:

  - `dataset`: Ultralytics-style dataset config (train/val paths, names, kpt_shape, ...)
  - `from_scratch` / `fine_tune` / `transfer` / `resume`: sections containing training params

See `data/train_example.yaml` for a template.
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# If Comet is enabled via env var, import it *before* torch/ultralytics so Comet can
# automatically hook into framework logging.
if os.environ.get("COMET_API_KEY"):
    # In some launchers (e.g. DDP wrappers), torch may already be imported before this
    # script executes. If so, Comet's auto-logging can't hook cleanly and will warn.
    # Disable Comet auto-logging in that case to silence the warning while still
    # allowing Ultralytics' Comet callbacks (manual logging) to run.
    if "torch" in sys.modules:
        os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
    try:
        import comet_ml  # noqa: F401
    except Exception:
        # If Comet isn't installed/usable, training should still proceed.
        pass

import torch
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import RUNS_DIR, SETTINGS


def _require_multilabel_ultralytics() -> None:
    ok = bool(getattr(ultralytics, "__multilabel__", False))
    assert ok, (
        "This repo requires an Ultralytics fork that supports multi-label detections.\n"
        "Install the multilabel fork/branch (see README)."
    )


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # keep stable, readable YAML; don't reorder keys
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def _get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def merge_weights_from(model_base: YOLO, merge_from: str) -> None:
    """
    Merge weights from `merge_from` into `model_base` (best-effort).

    - If name & shape match, copy full tensor.
    - If name matches but shape differs, copy overlapping slice.
    - If name doesn't exist in base, skip.
    """
    new_sd = YOLO(merge_from).model.state_dict()
    base_sd = model_base.model.state_dict()
    merged_sd: Dict[str, Any] = {}

    stats = {"full": 0, "partial": 0, "skipped_name": 0}
    total_seen = 0

    for name, new_param in new_sd.items():
        if name not in base_sd:
            stats["skipped_name"] += 1
            continue

        total_seen += 1
        base_param = base_sd[name]

        if new_param.shape == base_param.shape:
            merged_sd[name] = new_param.clone()
            stats["full"] += 1
        else:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(base_param.shape, new_param.shape))
            updated = base_param.clone()
            updated[slices] = new_param[slices]
            merged_sd[name] = updated
            stats["partial"] += 1
            print(f"[partial] {name}: copied region {slices}")

    model_base.model.load_state_dict(merged_sd, strict=False)

    print(f"Merging summary from `{merge_from}`:")
    print(f"  - Params found in both by name:  {total_seen}")
    print(f"    - fully replaced:   {stats['full']}")
    print(f"    - partially merged: {stats['partial']}")
    print(f"  - Params skipped (name missing in base): {stats['skipped_name']}")
    print("Merge complete.")


@dataclass(frozen=True)
class RunSpec:
    name: str
    project: str


def _default_run_spec(user_name: Optional[str], default_project: str) -> RunSpec:
    now = datetime.now()
    datestr = now.strftime("%Y%m%d")
    timestr = now.strftime("%H%M%S")
    base = user_name or f"yolo_dpar_{datestr}"
    name = f"{base}_{timestr}"
    return RunSpec(name=name, project=default_project)


def _pick_mode_section(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode not in cfg:
        raise KeyError(f"Config missing section '{mode}'. Expected one of: from_scratch, fine_tune, transfer, resume")
    section = cfg[mode] or {}
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{mode}' must be a mapping/dict")
    return section


def _write_dataset_tmp(dataset_cfg: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.yml"
    _dump_yaml(dataset_cfg, dataset_path)
    return dataset_path


def _make_model_source_from_scratch(train_cfg: Dict[str, Any], dataset_cfg: Dict[str, Any], out_dir: Path) -> str:
    """
    For from-scratch training, we often start from a model YAML (architecture),
    and need to set `nc` and sometimes `kpt_shape` based on the dataset.

    Expected keys in train_cfg:
      - config: path to model YAML
      - size: optional shorthand to inject into yolo{N}{size}* filenames (e.g. yolo11, yolo12, yolo26)
    """
    if "config" not in train_cfg:
        raise KeyError("from_scratch section must include 'config' (path to model YAML) or provide 'weights'")

    model_cfg_path = Path(str(train_cfg["config"]))
    model_cfg = _load_yaml(model_cfg_path)

    if "names" not in dataset_cfg:
        raise KeyError("dataset section must include 'names' (class names list) when training from scratch")

    model_cfg["nc"] = len(dataset_cfg["names"])
    if "kpt_shape" in model_cfg and "kpt_shape" in dataset_cfg:
        model_cfg["kpt_shape"] = dataset_cfg["kpt_shape"]

    model_name = model_cfg_path.name
    size = str(_get(train_cfg, "size", "")).strip()
    if size:
        # Inject or replace the model size letter after the "yolo<digits>" prefix.
        #
        # Examples:
        # - yolo26-pose.yaml + size=s -> yolo26s-pose.yaml
        # - yolo11l-pose.yaml + size=s -> yolo11s-pose.yaml
        # - yolo12n.yaml + size=n -> unchanged
        m = re.match(r"^(yolo\\d+)([nslmx])?(.*)$", model_name)
        if m:
            prefix, existing_size, rest = m.group(1), m.group(2), m.group(3)
            if existing_size:
                if existing_size != size:
                    model_name = f"{prefix}{size}{rest}"
            else:
                model_name = f"{prefix}{size}{rest}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model_yaml = out_dir / model_name
    print(f"Writing modified model YAML to {out_model_yaml}")
    _dump_yaml(model_cfg, out_model_yaml)
    return str(out_model_yaml)


def _resolve_model_source(
    mode: str,
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    run_dir: Path,
    default_project_dir: str | Path,
) -> str:
    """
    Resolve what to pass into `YOLO(...)`:
      - a `.pt` weights path, OR
      - a model YAML path for from-scratch builds.
    """
    if mode == "resume":
        # resume always needs weights
        if "weights" in train_cfg:
            return str(train_cfg["weights"])

        # Find latest run under project and use its last.pt.
        project = Path(str(_get(train_cfg, "project", default_project_dir)))
        candidates = [p for p in project.glob("*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"resume: no runs found under project folder '{project}'")
        latest = max(candidates, key=lambda p: p.stat().st_ctime)
        weights = latest / "weights" / "last.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"resume: expected weights at '{weights}' but file not found")
        print(f"Resuming from latest weights {weights}")
        return str(weights)

    if "weights" in train_cfg:
        return str(train_cfg["weights"])

    # from scratch: build a model YAML that matches dataset classes/keypoints
    return _make_model_source_from_scratch(train_cfg, dataset_cfg, run_dir)


def _auto_batch_and_device(model: YOLO, requested_batch: int, requested_device: str) -> tuple[int, Any]:
    """
    Ultralytics auto-batch doesn't work well for multi-GPU in some setups.
    Keep a simple heuristic from the original script.
    """
    if requested_device.strip():
        # user explicitly asked: trust them
        return requested_batch, requested_device

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return requested_batch, "cpu"

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    num_params_m = sum(p.numel() for p in model.model.parameters()) / 1_000_000.0

    if num_params_m < 4:
        model_size = "n"
    elif num_params_m < 15:
        model_size = "s"
    elif num_params_m < 35:
        model_size = "l"
    else:
        model_size = "x"

    batch = requested_batch
    device: Any = [0]

    if num_gpus > 1:
        if batch == -1:
            batch = 8 * num_gpus
            if gpu_gb > 20:
                batch = 16 * num_gpus
            if gpu_gb > 30:
                batch = 32 * num_gpus
            if model_size == "x":
                batch = int(batch / 2)
            elif model_size == "n":
                batch = int(batch * 3)
            elif model_size == "s":
                batch = int((batch * 3)//2)
        device = [x for x in range(num_gpus)]

    batch = int(batch)
    print(f"GPUs:{num_gpus} params:{num_params_m:.1f}M (size:{model_size}) gpu_GB:{gpu_gb:.1f} batch:{batch}")
    return batch, device


def main() -> None:
    _require_multilabel_ultralytics()

    parser = argparse.ArgumentParser(description="Train YOLO-DP/DPA/DPAR locally (no AzureML).")
    parser.add_argument("--config", type=str, default="data/train_example.yaml", help="training config YAML")
    parser.add_argument(
        "--mode",
        type=str,
        default="from_scratch",
        choices=["from_scratch", "fine_tune", "transfer", "resume"],
        help="which config section to use",
    )
    parser.add_argument("--name", type=str, default=None, help="run name (default: timestamped)")
    parser.add_argument("--project", type=str, default="runs", help="output folder for Ultralytics runs")
    parser.add_argument("--device", type=str, default="", help="Ultralytics device string (e.g. '0', '0,1', 'cpu')")
    parser.add_argument("--dry-run", action="store_true", help="print resolved config and exit")
    args = parser.parse_args()

    # Normalize `--project` into an absolute directory under Ultralytics' RUNS_DIR.
    #
    # If we pass a relative `project`, Ultralytics will nest it under `RUNS_DIR/<task>/<project>`,
    # e.g. RUNS_DIR/pose/runs/... which is often surprising.
    #
    # Expected:
    #   /.../ultralytics/runs/<run_name>
    # not:
    #   /.../ultralytics/runs/pose/runs/<run_name>
    if not Path(args.project).is_absolute():
        args.project = str(RUNS_DIR) if args.project == "runs" else str(Path(RUNS_DIR) / args.project)

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    cfg = _load_yaml(args.config)
    if "dataset" not in cfg or not isinstance(cfg["dataset"], dict):
        raise KeyError("Config must include a 'dataset' mapping (Ultralytics dataset config)")

    dataset_cfg = cfg["dataset"]
    train_cfg = _pick_mode_section(cfg, args.mode)

    run_spec = _default_run_spec(args.name, args.project)
    run_dir = Path(args.project) / run_spec.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Enable Comet logging iff COMET_API_KEY is set.
    #
    # Ultralytics' Comet callback uses:
    #   comet_ml.start(project_name=os.getenv("COMET_PROJECT_NAME", args.project))
    # We want the Comet *project* to match our run folder name, i.e. `run_spec.name`.
    if os.environ.get("COMET_API_KEY"):
        SETTINGS["comet"] = True
        os.environ["COMET_PROJECT_NAME"] = run_spec.name
        try:
            import comet_ml

            # `login()` is for auth/config; project is set later by `start(project_name=...)`.
            if hasattr(comet_ml, "login"):
                comet_ml.login()
        except Exception as e:
            print(f"[Comet] COMET_API_KEY set but Comet init failed; continuing without Comet. Error: {e}")

    # Write a per-run copy for provenance, but pass a stable path to Ultralytics.
    # Ultralytics resolves relative dataset paths relative to the dataset YAML location (yaml_file.parent),
    # so if we write a new YAML path every run, it can invalidate label caches and trigger rescans.
    _write_dataset_tmp(dataset_cfg, run_dir)

    dataset_path = Path("/tmp/dataset.yml")
    _dump_yaml(dataset_cfg, dataset_path)
    model_source = _resolve_model_source(args.mode, train_cfg, dataset_cfg, run_dir, default_project_dir=args.project)

    if args.dry_run:
        print("Resolved training run:")
        print(f"  config:   {args.config}")
        print(f"  mode:     {args.mode}")
        print(f"  project:  {args.project}")
        print(f"  name:     {run_spec.name}")
        print(f"  dataset:  {dataset_path} (stable path)")
        print(f"  model:    {model_source}")
        print("  train_cfg:")
        print(train_cfg)
        return

    model = YOLO(model_source)

    if "merge-weights" in train_cfg:
        merge_weights_from(model, str(train_cfg["merge-weights"]))

    # common train params (kept close to the original script)
    optimizer = _get(train_cfg, "optimizer", "auto")
    lr0 = _get(train_cfg, "lr0", 0.01)
    epochs = _get(train_cfg, "epochs", 50)
    imgsz = _get(train_cfg, "imgsz", 640)
    freeze = _get(train_cfg, "freeze", None)
    pose = _get(train_cfg, "pose", 0.25)
    attr = _get(train_cfg, "attr", 5.0)
    rle = _get(train_cfg, "rle", 1.0)  # RLE loss gain (pose tasks; ultralytics default 1.0)
    patience = _get(train_cfg, "patience", 200)
    batch = int(_get(train_cfg, "batch", -1))

    batch, device = _auto_batch_and_device(model, batch, args.device)

    # Let Ultralytics manage run folder naming via (project, name)
    train_kwargs: Dict[str, Any] = dict(
        data=str(dataset_path),
        project=args.project,
        name=run_spec.name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        pose=pose,
        attr=attr,
        rle=rle,
    )

    if freeze is not None:
        train_kwargs["freeze"] = freeze

    # resume is a special flag in Ultralytics
    if args.mode == "resume":
        train_kwargs = dict(data=str(dataset_path), resume=True)
    else:
        # for fine-tune / transfer / scratch, keep optimizer/lr0 explicit
        train_kwargs["optimizer"] = optimizer
        train_kwargs["lr0"] = lr0

    print("Training with kwargs:")
    for k in sorted(train_kwargs.keys()):
        print(f"  - {k}: {train_kwargs[k]}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
