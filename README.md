# YOLO26-DPARF

**One-pass YOLO26 models that jointly do detection + pose/keypoints + attributes + ReID embeddings + face image quality (FIQA)** *(also works with YOLO11/12)*.

These models are a proof-of-concept: the goal is to **collapse multiple downstream vision tasks into a single forward pass** with minimal extra compute/params, while keeping core detection quality close to a “normal” detector.

- This code, and the training code builds on the Ultralytics codebase and is subject to inherited license conditions from that project.
- Example weights were trained on a large number of datasets fused together and are subject to license conditions of those datasets.

| Example | Example |
|---|---|
| <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p0.png" width="420" alt="DPAR example 0"> | <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p1.png" width="420" alt="DPAR example 1"> |
| <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p2.png" width="420" alt="DPAR example 2"> | <img src="https://raw.githubusercontent.com/ubonpartners/yolo-dpar/main/images/p3.png" width="420" alt="DPAR example 3"> |

## Contents

- [What’s unique here](#whats-unique-here)
- [Model family (DP / DPA / DPAR)](#model-family-dp--dpa--dpar)
- [What the models output](#what-the-models-output)
- [Results and weights](#results-and-weights)
- [Quickstart (demo app)](#quickstart-demo-app)
- [Training (local, standalone)](#training-local-standalone)
- [How it works (multi-label per box)](#how-it-works-multi-label-per-box)
- [Related repositories](#related-repositories)
- [License](#license)

## What’s unique here

Most pipelines do: **detect → crop → run pose model → run attribute model → run ReID model → run face quality model**.

YOLO-DPAR instead aims for:

- **Single forward pass**: one model invocation produces boxes + keypoints + attribute scores + ReID vectors (+ FIQA).
- **Tiny overhead**: the added heads are designed to be light compared to running multiple separate networks.
- **Deployment simplicity**: fewer models to version, fewer pre/post steps, fewer latency spikes.

This is useful anywhere you want **rich metadata per person** in real time (analytics, safety, search, tracking/re-identification pipelines, and dataset mining).

## Model family (DP / DPA / DPAR)

The repo name “DPAR” reflects the full model. In practice you’ll see a small family:

- **DP**: Detection + pose/keypoints (person pose and face keypoints).
- **DPA**: DP + **binary attributes** per person.
- **DPAR**: DPA + **ReID embeddings** (+ **FIQA** for faces).

## What the models output

In the provided demo setup, the “full” models can produce:

- **Detection**: 5 base classes: `person`, `face`, `vehicle`, `animal`, `weapon`
- **Keypoints**
  - Faces: 5 points (2× eyes, nose, 2× mouth corners)
  - Persons: 17 points (COCO pose)
  - Optional **facepose** mode: 19 combined points on the person detection
- **Attributes (binary, on person boxes)**: a set of learned `person_*` attributes (listed below)
- **ReID embeddings**: a dedicated head outputs a ReID vector (default 192-d) for **person** detections
- **FIQA**: a face-quality score intended to capture effects like size/blur/occlusion/orientation

### Default binary attributes (`person_*`)

These are the default attribute labels/classes used by the DPA/DPAR variants:

- **Demographics**
  - `person_is_male`, `person_is_female`
  - `person_is_child`, `person_is_teen`, `person_is_adult`, `person_is_senior`

- **Face / head / appearance**
  - `person_is_wearing_hat_or_head_covering`
  - `person_is_wearing_a_mask_or_face_covering`
  - `person_wearing_glasses_or_sunglasses`
  - `person_has_facial_hair`
  - `person_has_shoulder_length_hair`
  - `person_has_buzz_cut_or_bald_head`

- **Accessories / carried items**
  - `person_has_bag_or_backpack`
  - `person_is_holding_mobile_phone`

- **Clothing (type / style)**
  - `person_is_wearing_a_uniform`
  - `person_has_long_sleeves`
  - `person_is_wearing_shorts`
  - `person_is_wearing_a_coat_or_jacket`
  - `person_is_wearing_hoodie`
  - `person_is_wearing_bright_colored_clothing`
  - `person_is_wearing_hi_vis_clothes`
  - `person_has_patterned_clothing`
  - `person_clothing_has_prominent_logo`

- **Body / pose / activity**
  - `person_has_heavy_build`
  - `person_is_lying_down`
  - `person_has_a_threatening_posture`
  - `person_is_running`
  - `person_is_fighting`
  - `person_is_behind_camera`

- **Weapons / safety**
  - `person_is_carrying_a_weapon`
  - `person_weapon_held_in_hands`

- **Visible tattoos**
  - `person_has_visible_tattoos`

- **Smoking**
  - `person_is_smoking_or_vaping`

- **Clothing color attributes**
  - **Top**
    - `person_top_is_white_or_light`
    - `person_top_is_black_or_gray_or_dark`
    - `person_top_is_blue_or_purple`
    - `person_top_is_green`
    - `person_top_is_red_or_pink`
    - `person_top_is_orange_or_beige_or_yellow`
  - **Bottom**
    - `person_bottom_is_white_or_light`
    - `person_bottom_is_black_or_gray_or_dark`
    - `person_bottom_is_blue_or_purple`
    - `person_bottom_is_green`
    - `person_bottom_is_red_or_pink`
    - `person_bottom_is_orange_or_beige_or_yellow`

- **FIQA threshold attributes** (derived from face quality score)
  - `person_fiqa_0.05`, `person_fiqa_0.25`, `person_fiqa_0.45`, `person_fiqa_0.65`, `person_fiqa_0.85`

Training notes:

- Models were trained on a ~350K-image mixed dataset built from COCO/OpenImages/Objects365/others, reprocessed with the dataset pipeline.
- Attribute labels were produced using a vision LLM (dataset config is not published; contact me if you need it).

## Results and weights

**YOLO26 mAP values, tables, and weights are in progress.** The results below are from the earlier YOLO11/12-based model family.

Evaluation details:

- Metrics were computed using `map.py` from the Dataset Processor repo.
- The “overall” values are a geometric mean over 5 validation sets (including COCO/OpenImages/Objects365 val splits).
- All results shown at 640×640.
- Not all model variants were trained for the same number of epochs (so comparisons aren’t perfectly “fair”).
- ReID is reported as Recall@K on a mixed set containing 399 IDs / 5736 person images.

| Model | Params (M) | mAP50 Person | mAP50 Face | mAP50 Vehicle | mAP50 Pose | mAP50 Face KP | mAP50 Attr (main) | mAP50 Attr (color) | ReID R@K (1,10) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Yolo11l-dpar-250525 | 26.5 | 0.864 | 0.833 | 0.699 | 0.827 | 0.779 | 0.588 | 0.569 | 0.476, 0.792 |
| [Yolo11l-dpar-210525](https://drive.google.com/file/d/15R2s1vqeKMmrqck7HMkzR7Bo_50z9e-o/view?usp=drive_link) | 26.5 | 0.856 | 0.823 | 0.695 | 0.826 | 0.779 | 0.585 | 0.556 | 0.256, 0.618 |
| Yolo11l-dpa-210525 | 26.2 | 0.856 | 0.823 | 0.695 | 0.826 | 0.779 | 0.585 | 0.556 | 0.035, 0.117 |
| [Yolo11l-dpa-131224](https://drive.google.com/file/d/1DwRpgS53MtQYM4G7Rm1K7OBxHhguaiI5/view?usp=drive_link) | 26.2 | 0.840 | 0.786 | 0.660 | 0.798 | 0.744 | 0.561 | 0.540 | — |
| [Yolo11l-dp-291024](https://drive.google.com/file/d/1veVJ9y6Set3oIDtZ47_Zpz6cnYqyMauy/view?usp=drive_link) | 26.2 | 0.849 | 0.854 | 0.648 | 0.790 | 0.779 | — | — | — |
| [yolo-dpa-s-21124](https://drive.google.com/file/d/1FUK6x26Z8Dz0gqw-20IHrvnUIKl8lLhk/view?usp=drive_link) | 10.1 | 0.818 | 0.755 | 0.556 | 0.757 | 0.741 | 0.500 | 0.477 | — |
| [yolo-dpa-n-251224](https://drive.google.com/file/d/1YDbFnwfd_xvlm4kkRiXCs_FMCPPOTfXP/view?usp=drive_link) | 3.0 | 0.779 | 0.682 | 0.443 | 0.692 | 0.683 | 0.431 | 0.437 | — |

Note: some newer weights exist but aren’t all published/added to the table yet.

## Quickstart (demo app)

This repo includes a small demo viewer that:

- opens a webcam or video file,
- runs a DP/DPA model,
- lets you click a person box to inspect attributes,
- auto-downloads weights if missing.

### Install

```bash
git clone git@github.com:ubonpartners/yolo-dpar.git
cd yolo-dpar

conda env create -f environment.yml
conda activate yolo-dpar

# missing from environment.yml (kept minimal on purpose)
pip install opencv-python
```

### Run

```bash
# webcam
python yolo-dpa-test.py --video webcam --model yolo-dpa-l

# or an mp4 file
python yolo-dpa-test.py --video /path/to/video.mp4 --model yolo-dpa-s
```

Supported demo models (auto-downloaded): `yolo-dpa-l`, `yolo-dp-l`, `yolo-dpa-s`, `yolo-dpa-n`.

Controls:

- Press `p` to pause.
- Left-click a person box to highlight it (and show attributes).

## Training (local, standalone)

This repo includes a standalone local training script: `train.py`.

It is adapted from the AzureML training code, but **removes Azure/MLflow dependencies** and runs purely locally via Ultralytics.

### Requirements

- You must use the **Ultralytics multilabel fork/branch** (see [Related repositories](#related-repositories)).
- You need a working PyTorch + CUDA (or CPU) environment appropriate for your machine.

### Config format

Training is driven by a single YAML file with this shape:

- `dataset`: an Ultralytics dataset config (train/val paths, `names`, optional `kpt_shape`, …)
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

## How it works (multi-label per box)

The base is an Ultralytics YOLO pose model (YOLO26 / YOLO11/12), but attributes create a key problem:

> Default YOLO postprocessing assumes **one class per box** (highest score wins), which is incompatible with “attributes” that should co-exist with `person`.

So a small Ultralytics fork is used to support multi-label behavior (needed both during training and at inference/postprocess time).

Two high-level inference strategies exist:

- **(A) Expand-then-merge**: emit one detection per class above threshold (so you might see `person`, `person_male`, `person_has_hat`, …), then merge them later.
- **(B) Fold-into-vector (preferred)**: suppress attribute-classes before NMS, then attach an attribute score vector to the surviving person detection.

The current demo app follows approach (A). The model changes required for training/inference live in the `multilabel` fork/branch.

ReID note:

- Unlike Ultralytics’ feature-copy hook, DPAR uses a dedicated head (PoseReID-style) and outputs `reid_embeddings` directly in the inference results (valid for `person` boxes).

## Related repositories

- [Ultralytics fork](https://github.com/ubonpartners/ultralytics/tree/multilabel) (required `multilabel` branch)
- [Dataset Processor](https://github.com/ubonpartners/dataset-processor) (dataset build + `map.py` evaluation)
- [AzureML tools](https://github.com/ubonpartners/azureml) (training helpers)
- [ReID adapter training](https://github.com/ubonpartners/reid) (adapter training + model fusion)
- [Track](https://github.com/ubonpartners/track) (tracking + evaluation toolkit)

## License

This work is dual-licensed:

- **AGPL** for **non-commercial use only**
- **Ubon Cooperative License**: <https://github.com/ubonpartners/license/blob/main/LICENSE>

The provided weights were trained using code derived from Ultralytics and are therefore subject to AGPL terms (and any additional dataset/model license terms).

Contact: [bernandocribbenza@gmail.com](mailto:bernandocribbenza@gmail.com?subject=yolo-dpar%20question)
