"""Helpers for converting Ultralytics YOLO Results to detection dicts and drawing them.

Detection dict schema
---------------------
  box          [x1, y1, x2, y2]  normalised 0-1
  class        int  class index
  confidence   float
  id           tracker ID or None
  face_points  flat [x, y, v, ...]   5 face keypoints drawn as circles
               order: R.Eye, L.Eye, Nose, R.Mouth, L.Mouth
               set for face detections from 5-kpt or combined 22-kpt models
  pose_points  flat [x, y, v, ...]  22-entry flat list, COCO indices 0-21
               set for person detections; head kpts 0-4 are zero (skipped in skeleton)
               body kpts 5-16: L/R Shoulder, Elbow, Wrist, Hip, Knee, Ankle
  attrs        list[float]  attribute scores 0-1        (attr-head models)
"""

# ---------------------------------------------------------------------------
# Results -> dicts
# ---------------------------------------------------------------------------

def yolo_results_to_dets(results, det_thr=0.01):
    """Convert a single Ultralytics Results object to a list of detection dicts.

    Supports:
    - Standard detection/pose results (boxes, optional keypoints)
    - Attribute head: result.attributes is [N, attr_nc] tensor (sigmoid scores)
    - Tracker IDs: result.boxes.id when a tracker is active
    """
    boxes   = results.boxes.xyxyn.tolist()
    classes = results.boxes.cls.tolist()
    confs   = results.boxes.conf.tolist()
    ids     = results.boxes.id.tolist() if results.boxes.id is not None else [None] * len(boxes)

    kp_xy   = results.keypoints.xyn.tolist()  if results.keypoints is not None else None
    kp_conf = (results.keypoints.conf.tolist()
               if results.keypoints is not None and results.keypoints.has_visible
               else None)

    # Attribute head: [N, attr_nc] float tensor, values are post-sigmoid (0-1).
    attr_list = results.attributes.tolist() if results.attributes is not None else None

    dets = []
    for i, (box, cls, conf, tid) in enumerate(zip(boxes, classes, confs, ids)):
        if conf < det_thr:
            continue
        det = {"box": box, "class": int(cls), "confidence": conf, "id": tid}

        if kp_xy is not None:
            fp, pp = _unpack_keypoints(kp_xy[i], kp_conf[i] if kp_conf else None)
            if fp is not None: det["face_points"] = fp
            if pp is not None: det["pose_points"] = pp

        if attr_list is not None:
            det["attrs"] = attr_list[i]

        dets.append(det)
    return dets


def _unpack_keypoints(kp_xy, kp_conf):
    """Flatten [x, y] pairs + confidences into the flat [x, y, v, ...] format.

    Returns (face_points, pose_points) -- at most one non-None.

    Model formats supported:
      5 kpts   -> face_points (circles); order: R.Eye, L.Eye, Nose, R.Mouth, L.Mouth
      17 kpts  -> pose_points (COCO skeleton)
      22 kpts  -> combined face/person model:
                  face detections  fill kpts 0-4 (face order), kpts 5-21 are zero
                  person detections fill kpts 5-18 (COCO body: L.Shoulder..R.Ankle + mouths)
                  Distinguished by whether any body kpt (5+) is visible.
    """
    n    = len(kp_xy)
    conf = kp_conf if kp_conf is not None else [1.0] * n
    flat = []
    for j in range(n):
        x, y = kp_xy[j]
        v = float(conf[j])
        if x <= 0 and y <= 0:
            v = 0.0
        flat.extend([x, y, v])

    if n == 5:
        return flat, None                   # standalone face model

    if n == 17:
        return None, flat                   # standalone pose model

    if n == 22:
        # Face detections: kpts 0-4 set, 5-21 zero.
        # Person detections: kpts 5-18 set, 0-4 zero.
        body_visible = any(flat[3*j+2] > 0 for j in range(5, 19))
        if body_visible:
            return None, flat               # person: full flat passed; _POSE_EDGES only
                                            # uses indices 0-16, head kpts 0-4 skip if v=0
        else:
            return flat[:15], None          # face: first 5 kpts as circles

    return None, None


# ---------------------------------------------------------------------------
# Point-picking helper
# ---------------------------------------------------------------------------

def find_closest_det(dets, x, y):
    """Return (index, dist^2) of the detection whose box contains (x, y) and whose
    centre is closest to it.  Returns (None, inf) if no box contains the point.
    """
    best_idx, best_dist = None, float("inf")
    for i, det in enumerate(dets):
        b = det["box"]
        if not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
            continue
        cx, cy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
        d = (x - cx) ** 2 + (y - cy) ** 2
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx, best_dist


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# COCO 17-kpt skeleton.  Each entry is [a, b] or [a, b, mid_c].
_POSE_EDGES = [
    [0, 1], [0, 2], [0, 5, 6],
    [1, 3], [2, 4],
    [5, 6], [5, 11], [6, 12], [11, 12],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 13], [13, 15], [12, 14], [14, 16],
]


def kp_line(display, kp, pts, thickness=2, clr="half_blue"):
    """Draw one skeleton edge between keypoints kp[pts[0]] -> kp[pts[1]].

    If pts has three elements the end-point is the midpoint of pts[1] and pts[2].
    Skips the edge if any involved keypoint has zero visibility.
    """
    a, b = pts[0], pts[1]
    c    = pts[2] if len(pts) > 2 else None
    x0, y0, v0 = kp[3*a], kp[3*a+1], kp[3*a+2]
    x1, y1, v1 = kp[3*b], kp[3*b+1], kp[3*b+2]
    if v0 == 0 or v1 == 0:
        return
    if c is not None:
        x2, y2, v2 = kp[3*c], kp[3*c+1], kp[3*c+2]
        if v2 == 0:
            return
        x1, y1 = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    display.draw_line([x0, y0], [x1, y1], clr, thickness=thickness)


def draw_pose(display, kp, thickness=2, clr="half_blue"):
    """Draw a COCO 17-kpt skeleton from a flat [x, y, v, ...] keypoint list."""
    for edge in _POSE_EDGES:
        kp_line(display, kp, edge, thickness=thickness, clr=clr)


def draw_boxes(display, dets, class_names=None, attr_names=None,
               highlight_index=None, alt_clr=False):
    """Draw detection boxes, keypoints, and attribute info onto display.

    Args:
        display:         Display instance (see display.py).
        dets:            List of detection dicts from yolo_results_to_dets().
        class_names:     Optional list mapping class index -> name string.
        attr_names:      Optional list of attribute name strings (aligns with det["attrs"]).
        highlight_index: Index of the selected detection; drawn in a distinct colour
                         with a detail panel showing box coords and top attributes.
        alt_clr:         Use cyan instead of green for non-highlighted boxes.
    """
    for idx, det in enumerate(dets):
        highlighted = idx == highlight_index
        clr       = "flashing_yellow" if highlighted else ("half_cyan" if alt_clr else "half_green")
        thickness = 4 if highlighted else 2

        display.draw_box(det["box"], clr=clr, thickness=thickness)

        cls_name = (class_names[det["class"]]
                    if class_names and det["class"] < len(class_names)
                    else f"cls_{det['class']}")
        display.draw_text(f"{cls_name} {det['confidence']:.2f}",
                          det["box"][0], det["box"][3])

        # 5 face keypoints as coloured dots (right-side points in yellow)
        fp = det.get("face_points")
        if fp:
            for i in range(len(fp) // 3):
                if fp[3*i+2] > 0:
                    dot_clr = "half_yellow" if i in (0, 3) else "half_red"
                    display.draw_circle([fp[3*i], fp[3*i+1]], radius=0.004, clr=dot_clr)

        pp = det.get("pose_points")
        if pp:
            draw_pose(display, pp, thickness=thickness)

    # Detail panel for the highlighted detection
    if highlight_index is not None and highlight_index < len(dets):
        det = dets[highlight_index]
        b   = det["box"]
        lines = [
            f"Class: {class_names[det['class']] if class_names else det['class']}",
            f"Box:   L:{b[0]:.3f} T:{b[1]:.3f} R:{b[2]:.3f} B:{b[3]:.3f}",
            f"       W:{b[2]-b[0]:.3f} H:{b[3]-b[1]:.3f}",
        ]
        if attr_names and "attrs" in det:
            ranked = sorted(enumerate(det["attrs"]), key=lambda x: x[1], reverse=True)
            for i, v in ranked:
                if v > 0.2:
                    lines.append(f"  {attr_names[i]}: {v:.2f}")
        display.draw_text("\n".join(lines), 0.05, 0.05, unmap=False, fontScale=0.5)
