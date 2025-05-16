import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Sequence, Optional, Callable, Any, Tuple, List
import numpy as np
import math

# Optional dependencies
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

def default_match(det, gt, context):
    return det.match_score(gt, context)

def match_greedy(dets, gts, mfn=default_match, mfn_context=None):
    n_gts=len(gts)
    n_dets=len(dets)

    if n_gts==0 or n_dets==0:
        return [], [], []

    gt_matched=[False]*n_gts
    out_det_index=[]
    out_gt_index=[]
    out_cost=[]
    for i,det in enumerate(dets):
        best_v=0
        best_match=None
        for j,gt in enumerate(gts):
            v=mfn(det,gt,mfn_context)
            if gt_matched[j] is False and v>best_v:
                best_v=v
                best_match=j
        if best_match is not None:
            gt_matched[best_match]=True
            out_det_index.append(i)
            out_gt_index.append(best_match)
            out_cost.append(best_v)
    return out_det_index, out_gt_index, out_cost

def match_lsa(dets, gts, mfn=default_match, mfn_context=None):
    n_gts=len(gts)
    n_dets=len(dets)

    if n_gts==0 or n_dets==0:
        return [], [], []

    costs = [[0 for x in range(n_dets)] for y in range(n_gts)]
    for ii in range(n_gts):
        for jj in range(n_dets):
            if dets[jj] is not None and gts[ii] is not None:
                costs[ii][jj]=mfn(dets[jj], gts[ii], mfn_context)
    costs = np.array(costs)
    gt_ind, det_ind = linear_sum_assignment(np.array(costs), maximize=True)
    match_costs = costs[gt_ind, det_ind]

    return det_ind, gt_ind, match_costs

#t_evals=0
#t_tot=0

def match_lsa2(
    dets: Sequence[Any],
    gts: Sequence[Any],
    mfn: Callable[[Any, Any, Any], float] = default_match,
    mfn_context: Any = None,
    partition_fn: Optional[Callable[[Any, Any], int]] = None,
    partition_context: Any = None,
    max_partitions: Optional[int] = None,
    match_method: str = 'greedy',
) -> Tuple[List[int], List[int], List[float]]:
    """
    Assign detections to ground‑truths using one of several method

    If partition_fn is provided, only (gt,det) pairs whose bitmasks intersect are considered.
    Otherwise all pairs are considered for non‑Hungarian methods.

    #

    Returns
    -------
    det_indices, gt_indices, match_costs
    """
    global t_evals, t_tot

    n_dets, n_gts = len(dets), len(gts)
    if n_dets == 0 or n_gts == 0:
        return [], [], []

    method = match_method.lower()
    if method == 'hungarian':
        if linear_sum_assignment is None:
            raise ImportError("SciPy is required for Hungarian matching")
        # build full cost matrix
        C = np.zeros((n_gts, n_dets), dtype=float)
        for i, gt in enumerate(gts):
            if gt is None: continue
            for j, det in enumerate(dets):
                if det is None: continue
                C[i, j] = mfn(det, gt, mfn_context)
        gt_idx, det_idx = linear_sum_assignment(C, maximize=True)
        costs = C[gt_idx, det_idx].tolist()
        return det_idx.tolist(), gt_idx.tolist(), costs

    # Otherwise: build edge list (possibly sparse via partition_fn)
    edges: List[Tuple[int,int,float]] = []

    evals=0
    if partition_fn is None:
        # full edges
        for i, gt in enumerate(gts):
            if gt is None: continue
            for j, det in enumerate(dets):
                if det is None: continue
                c = mfn(det, gt, mfn_context)
                evals+=1
                if c != 0.0:
                    edges.append((i, j, c))
    else:
        # sparse via partition bins
        det_masks = [partition_fn(d, partition_context) for d in dets]
        gt_masks  = [partition_fn(g, partition_context) for g in gts]
        if max_partitions is None:
            all_masks = det_masks + gt_masks
            max_bit = max((m.bit_length() for m in all_masks), default=0)
        else:
            max_bit = max_partitions

        # bin indices by bit
        det_bins = [[] for _ in range(max_bit)]
        gt_bins  = [[] for _ in range(max_bit)]
        for j, mask in enumerate(det_masks):
            m = mask
            while m:
                lsb = m & -m
                k   = lsb.bit_length() - 1
                if k < max_bit:
                    det_bins[k].append(j)
                m &= m - 1
        for i, mask in enumerate(gt_masks):
            m = mask
            while m:
                lsb = m & -m
                k   = lsb.bit_length() - 1
                if k < max_bit:
                    gt_bins[k].append(i)
                m &= m - 1

        # build edges with 2D seen array
        seen = [[False]*n_dets for _ in range(n_gts)]
        for k in range(max_bit):
            for i in gt_bins[k]:
                if gts[i] is None: continue
                for j in det_bins[k]:
                    if dets[j] is None or seen[i][j]:
                        continue
                    seen[i][j] = True
                    c = mfn(dets[j], gts[i], mfn_context)
                    evals+=1
                    if c != 0.0:
                        edges.append((i, j, c))

    #t_evals+=evals
    #t_tot+=(len(gts)*len(dets))
    #print(100.0*t_evals/(t_tot))

    # dispatch to matching sub‑routine
    if method == 'greedy':
        return _match_greedy(edges)
    elif method == 'greedy_multi_match':
        return _match_greedy_multi_match(edges)
    else:
        raise ValueError(f"Unknown match_method: {match_method}")


def _match_greedy(
    edges: List[Tuple[int,int,float]]
) -> Tuple[List[int], List[int], List[float]]:
    """Greedy max‑weight matching on (i,j,weight) edges."""
    edges.sort(key=lambda x: x[2], reverse=True)
    matched_gts, matched_dets = set(), set()
    det_idx, gt_idx, costs = [], [], []
    for i, j, w in edges:
        if i in matched_gts or j in matched_dets:
            continue
        matched_gts.add(i)
        matched_dets.add(j)
        gt_idx.append(i)
        det_idx.append(j)
        costs.append(w)
    return det_idx, gt_idx, costs

def _match_greedy_multi_match(
    edges: List[Tuple[int,int,float]]
) -> Tuple[List[int], List[int], List[float]]:
    """Greedy max‑weight matching on (i,j,weight) edges."""
    edges.sort(key=lambda x: x[2], reverse=True)
    matched_gts, matched_dets = set(), set()
    det_idx, gt_idx, costs = [], [], []
    for i, j, w in edges:
        if j in matched_dets:
            continue
        matched_gts.add(i)
        matched_dets.add(j)
        gt_idx.append(i)
        det_idx.append(j)
        costs.append(w)
    return det_idx, gt_idx, costs

def uniform_grid_partition(
    box,
    context: Optional[Tuple[int,int]] = None
) -> int:
    """
    Map o.box = (x0, y0, x1, y1) to a bitmask of which cells
    in an R x C uniform grid it intersects.

    Parameters
    ----------
    o : object
        Must have `o.box = (x0, y0, x1, y1)`, floats in [0,1].
    context : (rows, cols) or None
        Grid shape.  If None, defaults to (4, 8).

    Returns
    -------
    mask : int
        Bitmask with bit (r*cols + c) set iff
        the box overlaps cell at row r, col c.
    """
    # default to 4 rows, 16 cols
    rows, cols = context if context is not None else (4, 16)
    x0, y0, x1, y1 = box

    # compute cell spans
    start_col = int(math.floor(x0 * cols))
    end_col   = int(math.ceil (x1 * cols)) - 1
    start_row = int(math.floor(y0 * rows))
    end_row   = int(math.ceil (y1 * rows)) - 1

    # clamp indices
    start_col = max(0, min(cols-1, start_col))
    end_col   = max(0, min(cols-1, end_col))
    start_row = max(0, min(rows-1, start_row))
    end_row   = max(0, min(rows-1, end_row))

    # build bitmask
    col_mask=0
    for c in range(start_col, end_col + 1):
        col_mask |= (1 << c)
    mask = 0
    for r in range(start_row, end_row + 1):
        mask |= (col_mask << (r * cols))

    return mask
