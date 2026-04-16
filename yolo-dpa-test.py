"""YOLO detection/pose/attribute demo with live video or webcam.

Works with any Ultralytics YOLO model from the ubon branch:
  - yolo26 / yolo11 detection or pose models
  - models with the attribute head (attr_nc > 0)
  - fused PoseReID models (reid_embeddings shown in console)

Usage:
  python yolo-dpa-test.py --model yolo26n.pt
  python yolo-dpa-test.py --model /path/to/yolo-attr.pt --video /path/to/clip.mp4

Keys:
  p        pause / unpause
  click    highlight the clicked detection and show attribute scores
  Esc      quit
"""

import argparse
from pathlib import Path

import cv2
import ultralytics
import other.display as other_display
import other.ultralytics as other_ultralytics

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_git_lfs_pointer_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb") as f:
            head = f.read(256)
    except OSError:
        return False
    return head.startswith(LFS_POINTER_PREFIX)


def _raise_if_model_is_lfs_pointer(model_path: str) -> None:
    """Catch common Git LFS pointer mistakes with an actionable error."""
    path = Path(model_path).expanduser()
    if not path.exists() or not path.is_file():
        return
    if _is_git_lfs_pointer_file(path):
        raise RuntimeError(
            f"Model file is a Git LFS pointer, not real weights: {path}\n"
            "Install/fetch Git LFS weights, then re-run:\n"
            "  git lfs install\n"
            "  git lfs pull --include=\"models/*.pt\""
        )


def get_attr_names(model):
    """Read attribute names from the model checkpoint, or return an empty list."""
    # YOLO -> AutoBackend -> actual nn.Module; attr_names may live at any level.
    for obj in [model, getattr(model, "model", None),
                getattr(getattr(model, "model", None), "model", None)]:
        if obj is not None:
            names = getattr(obj, "attr_names", None)
            if names:
                return list(names)
    return []


def do_video(model_path, video):
    _raise_if_model_is_lfs_pointer(model_path)
    model = ultralytics.YOLO(model_path)
    class_names = [model.names[i] for i in range(len(model.names))]
    attr_names  = get_attr_names(model)

    print(f"Model task : {model.task}")
    print(f"Classes    : {len(class_names)}")
    if attr_names:
        print(f"Attributes : {attr_names}")

    if video == "webcam":
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Cannot access webcam"
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS,            30)
    else:
        cap = cv2.VideoCapture(video)
        assert cap.isOpened(), f"Cannot open video: {video}"
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video      : {w}x{h}  {fps} fps  {fc} frames")

    paused        = False
    highlight_pos = None
    frame         = None
    display       = other_display.Display(width=1600, height=900)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

        results = model(frame, conf=0.2, max_det=500, half=True, verbose=False)
        dets    = other_ultralytics.yolo_results_to_dets(results[0], det_thr=0.2)

        display.clear()
        highlight_index = None
        if highlight_pos is not None:
            highlight_index, _ = other_ultralytics.find_closest_det(dets, *highlight_pos)

        other_ultralytics.draw_boxes(display, dets,
                                     class_names=class_names,
                                     attr_names=attr_names,
                                     highlight_index=highlight_index)
        display.show(frame, title=f"{model_path}  —  {len(dets)} detections")

        for event in display.get_events(5):
            if event["key"] == "p":
                paused = not paused
            if event["lbutton"]:
                highlight_pos = [event["x"], event["y"]]

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/yolo26l-e2e-v10r-080426.pt",
                        help="Path to YOLO .pt model (default: models/yolo26l-e2e-v10r-080426.pt)")
    parser.add_argument("--video", default="webcam",
                        help="Video source: 'webcam' or path to an mp4 file")
    opt = parser.parse_args()
    do_video(opt.model, opt.video)
