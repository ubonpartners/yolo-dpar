# YOLO-DPAR

**One-pass YOLO26 models that jointly do detection + pose/keypoints + attributes + ReID embeddings + face image quality (FIQA)** *(also works with YOLO11/12)*.

Most pipelines do: **detect → crop → run pose model → run attribute model → run ReID model → run face quality model**. YOLO-DPAR collapses this into a **single forward pass** with minimal extra compute, so you get rich per-person metadata (identities, attributes, quality scores) with fewer models, less latency, and simpler deployment.

Built on top of [Ultralytics](https://github.com/ultralytics/ultralytics) with a small fork that adds multi-label attribute outputs and a ReID head.

| Example | Example |
|---|---|
| <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p0.png" width="420" alt="DPAR example 0"> | <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p1.png" width="420" alt="DPAR example 1"> |
| <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p2.png" width="420" alt="DPAR example 2"> | <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p3.png" width="420" alt="DPAR example 3"> |

## Contents

- [Model family](#model-family-dp--dpa--dpar)
- [What the models output](#what-the-models-output)
- [Results and weights](#results-and-weights)
- [How it works](#how-it-works)
- [Quickstart (demo app)](#quickstart-demo-app)
- [Training (local, standalone)](#training-local-standalone)
- [Related repositories](#related-repositories)
- [License](#license)

---

## Model family (DP / DPA / DPAR)

| Variant | What it adds |
|---------|-------------|
| **DP** | Detection + pose/keypoints (17-point COCO pose + 5-point face) |
| **DPA** | DP + binary attributes per person |
| **DPAR** | DPA + ReID embedding per person |
| **DPARF** | DPAR + FIQA face quality score |

---

## What the models output

In the full DPAR/DPARF setup, a single forward pass produces:

- **Detection**: 5 classes — `person`, `face`, `vehicle`, `animal`, `weapon`
- **Keypoints**
  - Faces: 5 points (2× eyes, nose, 2× mouth corners)
  - Persons: 17 points (COCO pose)
- **Attributes**: binary scores per person box (see list below)
- **ReID embeddings**: 80-d L2-normalized vector per person detection
- **FIQA**: face quality score in `[0, 1]` per face detection, capturing objective quality factors such as blur, pose, expression, occlusion, and illumination

### Default binary attributes (`person_*`)

- **Demographics**: `person_is_male`, `person_is_female`, `person_is_child`, `person_is_teen`, `person_is_adult`, `person_is_senior`

- **Face / head**: `person_is_wearing_hat_or_head_covering`, `person_is_wearing_a_mask_or_face_covering`, `person_wearing_glasses_or_sunglasses`, `person_has_facial_hair`, `person_has_shoulder_length_hair`, `person_has_buzz_cut_or_bald_head`

- **Accessories**: `person_has_bag_or_backpack`, `person_is_holding_mobile_phone`

- **Clothing type**: `person_is_wearing_a_uniform`, `person_has_long_sleeves`, `person_is_wearing_shorts`, `person_is_wearing_a_coat_or_jacket`, `person_is_wearing_hoodie`, `person_is_wearing_bright_colored_clothing`, `person_is_wearing_hi_vis_clothes`, `person_has_patterned_clothing`, `person_clothing_has_prominent_logo`

- **Body / pose / activity**: `person_has_heavy_build`, `person_is_lying_down`, `person_has_a_threatening_posture`, `person_is_running`, `person_is_fighting`, `person_is_behind_camera`

- **Weapons / safety**: `person_is_carrying_a_weapon`, `person_weapon_held_in_hands`

- **Visible tattoos / smoking**: `person_has_visible_tattoos`, `person_is_smoking_or_vaping`

- **Clothing color — top**: `person_top_is_white_or_light`, `person_top_is_black_or_gray_or_dark`, `person_top_is_blue_or_purple`, `person_top_is_green`, `person_top_is_red_or_pink`, `person_top_is_orange_or_beige_or_yellow`

- **Clothing color — bottom**: `person_bottom_is_white_or_light`, `person_bottom_is_black_or_gray_or_dark`, `person_bottom_is_blue_or_purple`, `person_bottom_is_green`, `person_bottom_is_red_or_pink`, `person_bottom_is_orange_or_beige_or_yellow`

---

## Results and weights

Evaluation notes:
- Metrics are a geometric mean over 11 validation sets (COCO, OpenImages, Objects365 splits + internal sets).
- All results at 640×640 input.
- "L" suffix on a metric means large-object subset only (e.g. PoseL = pose mAP for person boxes ≥ 96×96 px).
- The table includes an `End2End` column plus ReID `Rank-1` and `ReID mAP` columns.
- ReID scores are reported from the validation subset of the [Ubon synthetic-reid dataset](https://github.com/ubonpartners/synthetic-reid).

<img src="images/results_weights_table.png" width="1700" alt="YOLO-DPAR results and weights table with End2End column">

Regenerate this image from source data with:

```bash
python other/render_results_table.py
```

Stock models are standard Ultralytics YOLO trained on COCO (person + vehicle only; no face/pose/attribute). DPA/DPAR model weights are in `models/` (Git LFS).

**Note on YOLO26 model names**: `-e2e` denotes end-to-end training variants, and `-v10r` denotes variants that include the ReID branch.

---

## How it works

### Dataset pipeline

The quality of a multi-task model like this is almost entirely determined by the training data. These models were trained on a large **multi-source merged dataset** built with the [Dataset Processor](https://github.com/ubonpartners/dataset-processor) pipeline:

- **Source merging**: COCO, OpenImages, Objects365, and several internal datasets are merged with format normalization and label harmonization.
- **Dataset completion**: missing annotations are filled in automatically — e.g. a dataset that has person boxes but no face boxes gets face detections added by a strong face detector.
- **Hard example mining**: training is biased towards difficult or rare examples (crowded scenes, small objects, unusual poses) rather than just uniformly sampling the merged set.
- **Attribute labeling**: binary person attributes (see [full list above](#default-binary-attributes-person_)) were labeled automatically at scale using a **vision LLM**, making it practical to annotate millions of person instances with attributes that would be prohibitively expensive to label by hand.

The pipeline is config-driven and repeatable. Evaluation is via the `map.py` tool in Dataset Processor, which runs across all validation sets and reports a geometric mean.

### Attribute detection

Binary attributes are output by the model's classification head. In the published models, attribute logits are **packed alongside the main class scores** in the same detection head branch — the final 1×1 conv outputs `nc + attr_nc` channels rather than `nc` alone. This means attribute outputs come for free, sharing all the feature extraction work of the detection head.

**Overhead (yolo26l, nc=5, 45 attributes, 3 detection scales at c3=256):**

The only extra parameters are in the final output layer: 45 extra output channels × 257 (256 input + bias) × 3 scales ≈ **+35K parameters** (<0.2% of the 26.5M model). GFLOPs overhead is similarly negligible. Attributes add almost no cost because they reuse the same intermediate feature maps as the class outputs.

### ReID

The ReID path uses a separate **FiLM-modulated MLP adapter** (`ReIDAdapter`) trained on top of a frozen detector, then fused back in:

1. **Adapter training**: the detector's per-person feature vectors (aggregated across detection scales) are mapped to L2-normalized ReID embeddings using a small MLP. Only the adapter is trained; the backbone and detection head stay fixed. Metric-learning supervision over identity-labeled data shapes the embedding space so that the same person across different views and times lands close together.

2. **Fusion**: once the adapter is trained, the base pose head is promoted to a ReID-capable variant (`PoseReID`/`Pose26ReID`) and the adapter weights are injected. After fusion, the model outputs `reid_embeddings` in the same forward pass with no runtime overhead from the adapter wrapper.

**Overhead (yolo26l, default adapter: in_dim=575, hidden1=160, hidden2=192, emb=80):**

**+147,599 parameters**. The fused ReID head adds **~145,872 MACs (~0.292 MFLOPs)** to the forward pass — a fixed cost independent of how many persons are detected, since the head runs over the full feature map and the per-person vectors are read out during post-processing.

The default embedding dimension is **80-d**, L2-normalized. Cosine similarity between two vectors gives a re-identification score; typical thresholds are 0.4–0.6 depending on the application.

The ReID-capable models in this repository were trained with the [Ubon synthetic-reid dataset](https://github.com/ubonpartners/synthetic-reid).

<img src="images/reid_comparison_grid_3x2.png" width="1800" alt="ReID comparison grid for stock YOLO26L, DPA, and DPAR models">

Comparison on two validation queries from the Ubon synthetic-reid dataset:

- **Left column** uses stock YOLO26L `feats` as the ReID baseline (not one of the packaged checkpoints) and has the weakest retrieval quality.
- **Middle column (`yolo26l-v10r-240226.pt`)** is the non-E2E YOLO26L DPAR checkpoint and is significantly better.
- **Right column (`yolo26l-e2e-v10r-080426.pt`)** is the E2E YOLO26L DPAR checkpoint and gives the strongest retrieval quality across both queries.

In each panel, the **top-left cell is the query image**, and the remaining cells are the **top-15 matches by cosine similarity** on the ReID vector.
In the retrieval visualization, **green** boxes indicate correct matches and **blue** boxes indicate incorrect matches.

For the full adapter training and fusion workflow, see the companion [reid repo](https://github.com/ubonpartners/reid).

### FIQA

FIQA (Face Image Quality Assessment) estimates how suitable a detected face crop is for recognition — penalizing blur, extreme pose, partial occlusion, and small size. It is output as a single continuous score in [0, 1] per face detection, derived from a learned linear combination of the model's attribute weights.

In the tracking pipeline, FIQA gates which face crops are worth passing to the (more expensive) face embedding model: low-quality crops are skipped, which both improves embedding accuracy and reduces GPU load.

---

## Quickstart (demo app)

This repo includes a small demo viewer that opens a webcam or video file, runs a DPA/DPAR model, and lets you click a person box to inspect attributes.

**IMPORTANT: The demo app requires the `ubon26` branch from [https://github.com/ubonpartners/ultralytics](https://github.com/ubonpartners/ultralytics/tree/ubon26). If you install upstream `ultralytics` (or any other branch), the demo will not work.**

```bash
git clone git@github.com:ubonpartners/yolo-dpar.git
cd yolo-dpar

conda env create -f environment.yml
conda activate yolo-dpar

# required: install Ubon Ultralytics fork on branch ubon26
git clone https://github.com/ubonpartners/ultralytics.git ../ultralytics-ubon26
cd ../ultralytics-ubon26
git checkout ubon26
pip install -e .
cd ../yolo-dpar
```

Model weights are in `models/` via Git LFS. Ensure Git LFS is installed; if weights are still pointers after clone, run `git lfs pull`.

```bash
# webcam (default: models/yolo26l-e2e-v10r-080426.pt)
python yolo-dpa-test.py --video webcam

# video file
python yolo-dpa-test.py --video /path/to/video.mp4 --model models/yolo26s-e2e-v10-100426.pt
python yolo-dpa-test.py --video /path/to/video.mp4 --model models/yolo26n-e2e-v10-050426.pt  # fastest
python yolo-dpa-test.py --video /path/to/video.mp4 --model models/yolo26s-e2e-v10r-100426.pt  # with ReID
```

Controls: press `p` to pause; left-click a person box to highlight it and show attributes.

---

## Benchmarking models

This repo includes a benchmark runner that wraps Ultralytics `benchmark()` for DPA/DPAR checkpoints:

```bash
# quick smoke test on 1 model, 50 val images, PyTorch format only
python benchmark_models.py \
  --data /mldata/v10/coco/dataset.yaml \
  --model-glob "yolo26s-e2e-v10r-100426.pt" \
  --formats "-" \
  --smoke-val-images 50 \
  --half

# full run on all models with common export formats
python benchmark_models.py \
  --data /mldata/v10/coco/dataset.yaml \
  --formats "-,onnx,engine" \
  --half \
  --device 0
```

What this script does:
- Copies each `.pt` to a temporary directory before running benchmark/export.
- Applies a small runtime compatibility patch for `task=posereid`.
- Collects all benchmark outputs into:
  - `runs/benchmarks/<timestamp>/combined_results.csv`
  - `runs/benchmarks/<timestamp>/combined_results.json`
  - `runs/benchmarks/<timestamp>/combined_results_table.png`
- Preserves benchmark metrics as real columns (for example `metrics/mAP50-95(P)`), not generic name/value pairs.
- For `posereid` models, also adds:
   - `metrics/mAP50_person(B)` (box AP50 for class `person` only)
   - `metrics/mAP50_person(P)` (pose AP50 for class `person` only)
   - `metrics/mAP50_all(B)` (box AP50 over all classes)

Notes:
- Ultralytics benchmark still logs to console and writes `benchmarks.log`, but this wrapper also returns structured combined output.
- If your full validation set is large, use `--smoke-val-images` first to verify the pipeline and export dependencies quickly.

---

## Training (local, standalone)

`train.py` is a local training script that runs purely via Ultralytics — no cloud dependencies.

**Requirements**: use the [Ultralytics `ubon26` fork/branch](https://github.com/ubonpartners/ultralytics/tree/ubon26) and a working PyTorch + CUDA environment.

### Config format

Training is driven by a single YAML file:
- `dataset`: Ultralytics dataset config (train/val paths, `names`, optional `kpt_shape`, …)
- `from_scratch` / `fine_tune` / `transfer` / `resume`: training parameter blocks

See `data/train_example.yaml` for a ready-to-edit template.

### Commands

```bash
# from scratch (writes a run-local model YAML with nc/kpt_shape set)
python train.py --config data/train_example.yaml --mode from_scratch

# fine-tune from a checkpoint
python train.py --config data/train_example.yaml --mode fine_tune

# transfer learning (freeze early layers)
python train.py --config data/train_example.yaml --mode transfer

# resume from the latest run under ./runs/
python train.py --config data/train_example.yaml --mode resume

# dry-run to inspect resolved paths/kwargs
python train.py --config data/train_example.yaml --mode from_scratch --dry-run
```

---

## Related repositories

- [Ultralytics (upstream)](https://github.com/ultralytics/ultralytics)
- [Ultralytics fork](https://github.com/ubonpartners/ultralytics/tree/ubon26) — required `ubon26` branch with attribute/ReID head
- [Dataset Processor](https://github.com/ubonpartners/dataset-processor) — dataset build pipeline and `map.py` multi-set evaluation
- [ReID adapter training](https://github.com/ubonpartners/reid) — adapter training and model fusion
- [Synthetic ReID dataset generation](https://github.com/ubonpartners/synthetic-reid) — synthetic identity-collage generation used for ReID training
- [Track](https://github.com/ubonpartners/track) — tracking and evaluation toolkit

---

## License

This work is dual-licensed:

- **AGPL** for **non-commercial use only**
- **Ubon Cooperative License**: <https://github.com/ubonpartners/license/blob/main/LICENSE>

The provided weights were trained using code derived from Ultralytics (AGPL) and on datasets that carry their own license restrictions — including COCO, OpenImages, and Objects365, each of which imposes conditions on commercial use and redistribution of derivative models. **You must review and comply with the license terms of all training datasets before using these weights in any product or service.** This is your responsibility; it is not waived by the AGPL or Ubon Cooperative License terms above.

Contact: [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=yolo-dpar%20question)
