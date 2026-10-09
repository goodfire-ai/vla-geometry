"""Branch-query selection around the grasp, from a `BranchEngine.run_baseline` result (or a loaded
capture, which has the same per-query signals).

Signals per query q: `grip_series[q]` (executed gripper command, 0 = open ... 255 = closed;
"closed" means > 127), `grasp_q[q]` (gripper in positive-force contact with the object) and
`objz_q[q]` (object height).

Stable-grasp onset = first query where contact holds for K = 3 consecutive queries; if the object
is lifted by more than 2 cm anywhere in the episode, those K queries must also include a lift.

Landmarks (edit target: +1 = close, -1 = open):
  before_grasp  gripper open, before the onset   -> +1
  after_grasp   gripper closed and holding, after the onset -> -1
  start, end    the first and last query (-> +1 and -1)
"""
from __future__ import annotations

import numpy as np

CLOSE, OPEN = +1.0, -1.0


def stable_grasp_onset(grasp_q, objz_q=None, *, K: int = 3, lift_thresh: float = 0.02) -> int | None:
    """First query q with contact at q..q+K-1 (and, when the episode shows a lift anywhere, a lift
    > lift_thresh within those queries). None if the object is never held."""
    g = list(grasp_q)
    n = len(g)
    z0 = None
    if objz_q is not None and len(objz_q) and np.isfinite(objz_q[0]):
        z0 = float(objz_q[0])
    for q in range(n):
        win = g[q:q + K]
        if len(win) < K or not all(win):
            continue
        if z0 is not None and objz_q is not None:
            lifted = any(np.isfinite(objz_q[qq]) and (objz_q[qq] - z0) > lift_thresh
                         for qq in range(q, min(n, q + K)))
            if not lifted and any(np.isfinite(z) and (z - z0) > lift_thresh for z in objz_q):
                continue
        return q
    return None


def find_landmarks(baseline: dict, *, K: int = 3, pre: int = 3, post: int = 3,
                   grip_thresh: float = 127.0) -> dict:
    """{name: {"chunk": query, "step": env step, "target": +1/-1}} plus "_grasp_onset_query".

    `start` and `end` are always present. With a stable grasp:
      after_grasp  = `post` queries after the onset, clamped to the contiguous closed + holding run
                     that starts at the onset (or the first closed + holding query at or after it);
      before_grasp = `pre` queries before the onset if the gripper is open there, else the nearest
                     open query before the onset (ties go to the later query)."""
    qs = list(baseline["query_steps"])
    nq = len(qs)
    out: dict = {}
    if nq == 0:
        return out

    out["start"] = {"chunk": 0, "step": qs[0], "target": CLOSE}
    out["end"] = {"chunk": nq - 1, "step": qs[-1], "target": OPEN}

    grip = list(baseline.get("grip_series", []))
    grasp = list(baseline.get("grasp_q", []))
    closed = [bool(i < len(grip) and grip[i] is not None and float(grip[i]) > grip_thresh)
              for i in range(nq)]
    held = [bool(i < len(grasp) and grasp[i]) for i in range(nq)]

    onset = stable_grasp_onset(grasp, baseline.get("objz_q"), K=K)
    if onset is None:
        return out

    run_end = onset
    while run_end + 1 < nq and closed[run_end + 1] and held[run_end + 1]:
        run_end += 1
    aq = min(onset + post, run_end)
    if not (closed[aq] and held[aq]):
        aq = next((q for q in range(onset, nq) if closed[q] and held[q]), None)

    target_b = onset - pre
    if 0 <= target_b < nq and not closed[target_b]:
        bq = target_b
    else:
        opens = [q for q in range(nq) if q < onset and not closed[q]]
        bq = min(opens, key=lambda q: (abs(q - target_b), -q)) if opens else None

    if bq is not None:
        out["before_grasp"] = {"chunk": bq, "step": qs[bq], "target": CLOSE}
    if aq is not None:
        out["after_grasp"] = {"chunk": aq, "step": qs[aq], "target": OPEN}
    out["_grasp_onset_query"] = onset
    return out


def _grip_state(baseline: dict, *, grip_thresh: float = 127.0):
    """(nq, closed[q], held[q], onset) from the baseline signals."""
    nq = len(baseline["query_steps"])
    grip = list(baseline.get("grip_series", []))
    grasp = list(baseline.get("grasp_q", []))
    closed = [bool(i < len(grip) and grip[i] is not None and float(grip[i]) > grip_thresh)
              for i in range(nq)]
    held = [bool(i < len(grasp) and grasp[i]) for i in range(nq)]
    onset = stable_grasp_onset(grasp, baseline.get("objz_q"))
    return nq, closed, held, onset


def _spread(cands: list[int], k: int) -> list[int]:
    """Up to `k` entries of the sorted list `cands`, evenly spread and including both ends."""
    if k <= 0 or not cands:
        return []
    if len(cands) <= k:
        return list(cands)
    pos = np.linspace(0, len(cands) - 1, k)
    idx = sorted({int(round(p)) for p in pos})
    i = 0
    while len(idx) < k and i < len(cands):
        if i not in idx:
            idx.append(i)
            idx = sorted(set(idx))
        i += 1
    return [cands[j] for j in sorted(idx)[:k]]


def find_landmark_windows(baseline: dict, *, w_pre: int = 3, w_post: int = 3,
                          buffer: int = 3, span: int = 6, grip_thresh: float = 127.0) -> dict:
    """{"before_grasp": [q, ...], "after_grasp": [q, ...]}: several branch queries per grasp
    landmark, spread evenly over queries at offsets buffer..buffer+span from the onset.

      before_grasp: up to `w_pre` gripper-open queries q with buffer <= onset - q <= buffer + span
                    (if none, any open query before the onset)
      after_grasp:  up to `w_post` closed + holding queries q with buffer <= q - onset <= buffer + span,
                    within the contiguous closed + holding run from the first such query at or after
                    the onset (if none, any query of that run)

    Empty dict if there is no stable grasp."""
    nq, closed, held, onset = _grip_state(baseline, grip_thresh=grip_thresh)
    out: dict = {}
    if nq == 0 or onset is None:
        return out

    pre_all = [q for q in range(0, onset) if not closed[q]]
    pre_win = [q for q in pre_all if buffer <= onset - q <= buffer + span]
    pre = _spread(pre_win or pre_all, w_pre)
    if pre:
        out["before_grasp"] = pre

    aq0 = next((q for q in range(onset, nq) if closed[q] and held[q]), None)
    if aq0 is not None:
        run_end = aq0
        while run_end + 1 < nq and closed[run_end + 1] and held[run_end + 1]:
            run_end += 1
        post_all = list(range(aq0, run_end + 1))
        post_win = [q for q in post_all if buffer <= q - onset <= buffer + span]
        post = _spread(post_win or post_all, w_post)
        if post:
            out["after_grasp"] = post
    return out
