"""
Standalone local training entrypoint for YOLO-DP/DPA/DPAR models.

This is adapted from `azureml/run_train.py` but intentionally removes:
- AzureML submission / datastores / mounts
- MLflow experiment wiring

It keeps the original "one YAML config drives everything" pattern:

  - `dataset`: Ultralytics-style dataset config (train/val paths, names, kpt_shape, ...)
  - `from_scratch` / `fine_tune` / `transfer` / `resume`: sections containing training params
  - `end2end`: optional (true/false, default false) in mode sections; controls end-to-end
    NMS-free detection head (YOLO26/YOLOv10 style). In from_scratch, written to model YAML.

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

import warnings

import torch
import torch.nn as nn
import yaml

# Allow TF32 for fp32 matmul operations (Linear layers, loss computations, optimizer step).
# cuDNN convolutions already use TF32 by default; this extends it to matmul.
# Negligible precision impact for training; meaningful throughput gain on Ampere/Blackwell.
torch.set_float32_matmul_precision("high")

# Suppress FutureWarning spam from torch._inductor internals during torch.compile.
# torch/_inductor/lowering.py still uses the deprecated torch._prims_common.check API;
# this will be fixed in a future PyTorch release.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\._inductor")
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


def _guess_base_model_name(model_source: str) -> Optional[str]:
    """Extract base model name from model source, e.g. 'yolo26n-dpar-v10.yaml' → 'yolo26n.pt'."""
    stem = Path(model_source).stem
    m = re.match(r"^(yolo\d+[nslmx])", stem)
    return (m.group(1) + ".pt") if m else None


def init_one2one_from_one2many(model: YOLO) -> None:
    """Bootstrap one2one head weights from the model's own one2many counterparts.

    The one2one_cv4* modules are deep-copied from cv4* at yaml-build time, BEFORE
    merge_weights_from runs.  So after merging a trained pose checkpoint, cv4_kpts and
    cv4_sigma carry good pose-trained weights while one2one_cv4_kpts/sigma are still at
    random yaml init.  This function copies those weights across so that freeze-based
    training doesn't start with a randomly-initialised one2one keypoint head.

    Call this BEFORE init_one2one_from so that the e2e-specific weights loaded from an
    external source take precedence where shapes match.
    """
    dst_head = model.model.model[-1]
    one2one_attrs = [a for a in dst_head._modules if a.startswith("one2one_")]
    copied = skipped = 0
    for attr in one2one_attrs:
        base_attr = attr[len("one2one_"):]
        base_mod = getattr(dst_head, base_attr, None)
        if base_mod is None:
            continue
        dst_mod = getattr(dst_head, attr)
        base_sd = base_mod.state_dict()
        dst_sd = dst_mod.state_dict()
        update: Dict[str, Any] = {}
        for k, dst_val in dst_sd.items():
            if k in base_sd and base_sd[k].shape == dst_val.shape:
                update[k] = base_sd[k].clone()
                copied += 1
            else:
                skipped += 1
        if update:
            dst_mod.load_state_dict(update, strict=False)
            print(f"  [o2o←o2m] {attr} ← {base_attr}: {len(update)} tensors")
    print(f"[init_one2one_from_one2many] {copied} tensors bootstrapped, {skipped} skipped (shape mismatch)")


def init_one2one_from(model: YOLO, source: str, only_attrs: list[str] | None = None) -> list[str]:
    """Copy one2one head weights from a pretrained E2E model.

    Exact-shape matches are copied in full.  Size mismatches (e.g. n vs l model) are
    handled with a partial copy of the overlapping slice so no weight is left at random
    init when a source tensor exists.

    Args:
        only_attrs: if given, restrict to this subset of one2one_* attribute names.
                    Used for fallback passes that only need to fill in missing modules.
    Returns:
        List of one2one_* attribute names that were absent from the source model.
    """
    print(f"[init_one2one] Loading source model: {source}")
    src_head = YOLO(source).model.model[-1]
    dst_head = model.model.model[-1]

    # Find all one2one_* nn.Module attributes on the destination head.
    # PyTorch stores submodules in _modules, not __dict__, so vars() misses them.
    one2one_attrs = [a for a in dst_head._modules if a.startswith("one2one_")]
    if only_attrs is not None:
        one2one_attrs = [a for a in one2one_attrs if a in only_attrs]
    if not one2one_attrs:
        print("[init_one2one] No one2one_* submodules found on destination head; nothing to do.")
        return []

    copied = skipped_shape = skipped_missing = 0
    missing_attrs: list[str] = []
    for attr in one2one_attrs:
        dst_mod = getattr(dst_head, attr)
        src_mod = getattr(src_head, attr, None)
        if src_mod is None:
            print(f"  [skip-missing] {attr}: not present in source head")
            skipped_missing += 1
            missing_attrs.append(attr)
            continue

        dst_sd = dst_mod.state_dict()
        src_sd = src_mod.state_dict()
        update: Dict[str, Any] = {}
        mod_copied = mod_partial = mod_skipped = 0
        for k, dst_val in dst_sd.items():
            if k not in src_sd:
                skipped_missing += 1
                mod_skipped += 1
            elif src_sd[k].shape == dst_val.shape:
                update[k] = src_sd[k].clone()
                copied += 1
                mod_copied += 1
            else:
                # Partial copy: fill overlapping slice, keep destination values elsewhere.
                # Handles n↔l size mismatches where channel counts differ.
                src_val = src_sd[k]
                slices = tuple(slice(0, min(a, b)) for a, b in zip(dst_val.shape, src_val.shape))
                merged = dst_val.clone()
                merged[slices] = src_val[slices]
                update[k] = merged
                skipped_shape += 1
                mod_partial += 1
        dst_mod.load_state_dict(update, strict=False)
        parts = [f"{mod_copied} copied"]
        if mod_partial:
            parts.append(f"{mod_partial} partial")
        if mod_skipped:
            parts.append(f"{mod_skipped} missing")
        print(f"  {attr}: {', '.join(parts)}")

    print(f"[init_one2one] Total: {copied} tensors copied, {skipped_shape} partial, {skipped_missing} missing")
    return missing_attrs


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
      - size: optional shorthand that controls the *filename* passed to Ultralytics (n/s/m/l/x).
              Ultralytics infers model scale from the size letter in the YAML filename.
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

    # end2end: whether to use end-to-end detection head (YOLO26, YOLOv10 style).
    # Set in model YAML so Ultralytics builds the correct architecture (one2one matcher, E2ELoss).
    model_cfg["end2end"] = bool(_get(train_cfg, "end2end", False))

    # Match AzureML `run_train.py` behavior:
    # - Load `config:` YAML contents
    # - Write a run-local copy whose *filename* includes the size letter (if provided)
    #   so Ultralytics can infer scale via `guess_model_scale()`.
    model_name = model_cfg_path.name
    size = str(_get(train_cfg, "size", "")).strip().lower()
    if size:
        if size not in {"n", "s", "m", "l", "x"}:
            raise ValueError(f"Invalid from_scratch.size '{size}'. Expected one of: n, s, m, l, x")

        # Inject or replace the model size letter after the "yolo<digits>" prefix.
        #
        # IMPORTANT: This must match Ultralytics' `guess_model_scale()` expectations, e.g.
        #   yolo11l-pose.yaml -> scale 'l'
        m = re.match(r"^(yolo\d+)([nslmx])?(.*)$", model_name)
        if m:
            prefix, existing_size, rest = m.group(1), m.group(2), m.group(3)
            if existing_size:
                if existing_size != size:
                    model_name = f"{prefix}{size}{rest}"
            else:
                model_name = f"{prefix}{size}{rest}"
        else:
            print(
                f"[warn] from_scratch.size='{size}' was set, but could not infer where to inject it "
                f"from config filename '{model_cfg_path.name}'. Ultralytics may default to the first scale."
            )

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

        # Find the newest actual checkpoint under project.
        # Using run-dir timestamps is fragile because this script creates a new run folder
        # before resolving resume source, which can make "latest run" point to an empty dir.
        project = Path(str(_get(train_cfg, "project", default_project_dir))).expanduser()
        checkpoints = [p for p in project.glob("*/weights/last.pt") if p.is_file()]
        if not checkpoints:
            raise FileNotFoundError(
                f"resume: no checkpoints found under '{project}'. "
                "Expected at least one '<run>/weights/last.pt'."
            )
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Resuming from latest checkpoint {latest}")
        return str(latest)

    if "weights" in train_cfg:
        return str(train_cfg["weights"])

    # from scratch: build a model YAML that matches dataset classes/keypoints
    return _make_model_source_from_scratch(train_cfg, dataset_cfg, run_dir)


def _apply_resume_overrides_to_checkpoint(weights_path: str | Path, train_cfg: Dict[str, Any], run_dir: Path) -> str:
    """
    If resume config includes train-arg overrides (pose/rle/attr/etc), patch them into
    checkpoint metadata and return the patched checkpoint path.

    We intentionally do not mutate the source checkpoint in-place.
    """
    control_keys = {"weights", "project"}
    overrides = {k: v for k, v in train_cfg.items() if k not in control_keys}
    if not overrides:
        return str(weights_path)

    src = Path(weights_path)
    if not src.is_file():
        raise FileNotFoundError(f"resume: checkpoint not found at '{src}'")

    ckpt = torch.load(src, map_location="cpu")

    patched_fields = 0
    for field in ("train_args", "args"):
        payload = ckpt.get(field)
        if isinstance(payload, dict):
            payload.update(overrides)
            patched_fields += 1

    # Some checkpoints may only have one of the above. Ensure train_args exists at minimum.
    if not isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"] = dict(overrides)
        patched_fields += 1

    out = run_dir / f"{src.stem}_resume_overrides.pt"
    torch.save(ckpt, out)
    print(
        f"resume overrides applied to checkpoint metadata ({patched_fields} field(s)): "
        f"{sorted(overrides.keys())}\n"
        f"  source: {src}\n"
        f"  patched: {out}"
    )
    return str(out)


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
                batch = int(batch * 2)
            elif model_size == "s":
                batch = int((batch * 3)//2)
        device = [x for x in range(num_gpus)]

    batch = int(batch)
    print(f"GPUs:{num_gpus} params:{num_params_m:.1f}M (size:{model_size}) gpu_GB:{gpu_gb:.1f} batch:{batch}")
    return batch, device


def main() -> None:
    #_require_multilabel_ultralytics()

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

    dataset_cfg = cfg["dataset"].copy()
    # If dataset has a path to a dir, merge in attributes/attr_nc/attr_label_format from dataset.yaml there (v10 format)
    path = dataset_cfg.get("path")
    if path and Path(path).is_dir():
        for name in ("dataset.yaml", "dataset.yml"):
            candidate = Path(path) / name
            if candidate.exists():
                from_file = _load_yaml(candidate)
                for k in ("attributes", "attr_nc", "attr_label_format", "attr_names"):
                    if k in from_file and k not in dataset_cfg:
                        dataset_cfg[k] = from_file[k]
                break
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

    if args.mode == "resume":
        model_source = _apply_resume_overrides_to_checkpoint(model_source, train_cfg, run_dir)

    model = YOLO(model_source)

    # When loading from weights (fine_tune/transfer/resume), optionally override end2end
    # from the config. For from_scratch, end2end is already set in the model YAML.
    if args.mode in ("fine_tune", "transfer", "resume") and "end2end" in train_cfg:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "end2end"):
            inner.end2end = bool(train_cfg["end2end"])
            print(f"Overriding model end2end to {inner.end2end} (from config)")

    if "merge-weights" in train_cfg:
        merge_weights_from(model, str(train_cfg["merge-weights"]))
        # Bootstrap one2one keypoint heads from the now-merged one2many weights.
        # The one2one_cv4* modules are deep-copied at yaml-build time (before merge),
        # so without this step they stay at random init even after a good merge.
        init_one2one_from_one2many(model)

    init_one2one = _get(train_cfg, "init_one2one", False)
    if init_one2one and freeze is not None:
        # When the backbone/neck is frozen the one2one head MUST be initialised from
        # weights that were trained on *this* backbone's feature distribution.
        # Weights from an external e2e model were calibrated to a different backbone;
        # loading them here would leave the one2one head permanently miscalibrated
        # (the frozen backbone can never adapt to close the gap), causing mAP to stay
        # at zero even as losses slowly decrease.
        # init_one2one_from_one2many (already called above) is the correct init: it
        # bootstraps the one2one head from the one2many head that was trained on these
        # exact backbone features.
        print(
            "[init_one2one] Skipping external e2e init because freeze is set.\n"
            "  The one2one head was already bootstrapped from one2many (compatible with frozen backbone).\n"
            "  An external e2e model's head is calibrated to a different backbone and would "
            "cause mAP≈0 with a frozen backbone."
        )
    elif init_one2one:
        if isinstance(init_one2one, str):
            # Primary pass: explicit trained e2e model. Partial copy handles size mismatches
            # (n↔l etc.) so every tensor gets a trained value, not random init.
            missing = init_one2one_from(model, init_one2one)
            # Fallback pass: auto-guessed base model (e.g. yolo26n.pt) only for modules
            # that were completely absent from the primary source.
            if missing:
                fallback_src = _guess_base_model_name(model_source)
                if fallback_src:
                    print(f"[init_one2one] Fallback pass for {len(missing)} missing module(s): {fallback_src}")
                    init_one2one_from(model, fallback_src, only_attrs=missing)
        else:
            one2one_src = _guess_base_model_name(model_source)
            if one2one_src is None:
                print(
                    f"[warn] init_one2one=true but could not auto-detect base model from '{model_source}'; "
                    "set init_one2one to an explicit model name (e.g. 'yolo26n.pt') to override."
                )
            if one2one_src:
                # Overrides the bootstrapped weights wherever the e2e source has better matches.
                init_one2one_from(model, one2one_src)

    # common train params (kept close to the original script)
    optimizer = _get(train_cfg, "optimizer", "auto")
    lr0 = _get(train_cfg, "lr0", 0.01)
    epochs = _get(train_cfg, "epochs", 50)
    imgsz = _get(train_cfg, "imgsz", 640)
    freeze = _get(train_cfg, "freeze", None)
    if freeze == "backbone":
        # Layers 0-10 are the backbone (Conv→C3k2→SPPF→C2PSA) for all YOLO11/26 sizes.
        # Ultralytics freezes layers 0..N-1 when given an int N, so 11 freezes the full backbone.
        freeze = 11
        print("freeze='backbone' → freezing layers 0-10 (backbone)")
    elif freeze == "backbone+neck":
        # Layers 0-22 are backbone + neck; layer 23 is the head (Pose/PoseReID/Detect).
        # Useful when fine-tuning only the head, e.g. initialising an E2E model from a
        # non-E2E checkpoint via merge-weights (one2one branches start from random init).
        freeze = 23
        print("freeze='backbone+neck' → freezing layers 0-22 (backbone + neck)")
    pose = _get(train_cfg, "pose", 0.25)
    attr = _get(train_cfg, "attr", 5.0)
    rle = _get(train_cfg, "rle", 1.0)  # RLE loss gain (pose tasks; ultralytics default 1.0)
    compile_mode = _get(train_cfg, "compile", False)  # torch.compile: False, True, or mode string
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
        compile=compile_mode,
    )

    train_kwargs["warmup_bias_lr"] = 0.0

    if "end2end" in train_cfg:
        train_kwargs["end2end"] = bool(train_cfg["end2end"])

    if freeze is not None:
        train_kwargs["freeze"] = freeze

    backbone_lr_scale = _get(train_cfg, "backbone_lr_scale", None)
    if backbone_lr_scale is not None:
        backbone_lr_scale = float(backbone_lr_scale)
        train_kwargs["backbone_lr_scale"] = backbone_lr_scale
        if backbone_lr_scale == 0.0:
            print(
                f"[warn] backbone_lr_scale=0 freezes backbone weights but still computes gradients (wasteful). "
                "Use 'freeze: backbone+neck' instead for efficiency."
            )
        else:
            print(f"backbone_lr_scale={backbone_lr_scale} — backbone/neck layers will use lr * {backbone_lr_scale}")

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
