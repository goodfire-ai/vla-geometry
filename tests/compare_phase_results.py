"""Compare the Figure 5 analysis results with reference result files.

  scan     phase/decoder.json (stage 30) against a reference per-layer LDA scan: per layer the
           balanced accuracy (lda_func_bal), raw accuracy (lda_func_raw), majority rate (maj),
           per-phase recall and confusion matrix, and the best layer
  decode   phase/freeze/decode_ae.json (stage 50) against a reference freeze-test fit: episode
           and landmark counts, layer, the composition per landmark kind and condition
           (decode_natural_composition) and the decoder record (natural_decoder_info: rows,
           houses, balanced accuracy, fold hash)
  gripper  phase/fig5_numbers.json (stage 70, panel d) against a reference gripper summary:
           episode count, executed steps and per phase n, mean and 95% CI, and the closed fraction
           per executed step, natural and frozen (per_panel)
  slope    phase/fig5_numbers.json (stage 70, panel c) against a reference panel c summary:
           landmark count, n per phase, % decoded correctly natural and frozen, and the two most
           frequent decoded phases when frozen with their % (the reference stores % rounded to
           2 decimals; the new values are rounded the same way before comparing)
  counts   phase/fig5_numbers.json (stage 70, panel d) against a reference per-kind action
           summary: episode count, landmark count and landmarks per phase (grasp = open- plus
           closed-hand grasp, place = closed-hand place)

Values are compared exactly (floats bit for bit, apart from the rounding of slope).

  python tests/compare_phase_results.py scan --new <decoder.json> --ref <reference scan .json>
  python tests/compare_phase_results.py decode --new <decode_ae.json> --ref <reference fit .json>
  python tests/compare_phase_results.py gripper --new <fig5_numbers.json> --ref <reference .json>
  python tests/compare_phase_results.py slope --new <fig5_numbers.json> --ref <reference .json>
  python tests/compare_phase_results.py counts --new <fig5_numbers.json> --ref <reference .json>

Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += _walk(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}/{k}")
        return out
    return [] if a == b else [(path, a, b)]


def scan_pairs(new: dict, ref: dict):
    names = ["reach", "grasp", "transport", "place"]
    pairs = {"best_layer": (new["best_layer"], ref["best_layer"])}
    for L, r in ref["per_layer"].items():
        n = new["per_layer"].get(L)
        if n is None:
            pairs[f"layer {L}"] = ("<missing>", "present")
            continue
        pairs[f"layer {L}"] = (
            {"bal": n["bal"], "raw": n["raw"], "maj": n["maj"], "confusion": n["confusion"],
             "recall": {k: n["recall"][k] for k in names}},
            {"bal": r["lda_func_bal"], "raw": r["lda_func_raw"], "maj": r["maj"],
             "confusion": r["confusion"], "recall": {k: r["recall"][k] for k in names}})
    return pairs


def decode_pairs(new: dict, ref: dict):
    keys = ("n_natural_rows", "n_houses", "natural_balanced_acc", "fold_hash")
    return {"n_files": (new["n_files"], ref["n_files"]), "n_rows": (new["n_rows"], ref["n_rows"]),
            "layer": (new["layer"], ref["layer"]),
            "composition": (new["composition"], ref["decode_natural_composition"]),
            "decoder": ({k: new["decoder"][k] for k in keys},
                        {k: ref["natural_decoder_info"][k] for k in keys})}


def gripper_pairs(new: dict, ref: dict):
    d = new["d"]
    pairs = {"n_episodes": (d["n_episodes"], ref["n_files"]),
             "executed_steps": (d["executed_steps"], ref["executed_steps"])}
    fields = {"n": "n", "delta_closed_mean": "delta_closed_mean",
              "delta_closed_ci95": "delta_closed_ci95", "closed_natural": "closed_frac_natural",
              "closed_frozen": "closed_frac_frozen"}
    for k, r in ref["per_panel"].items():
        n = d["per_phase"].get(k, {})
        pairs[k] = ({f: n.get(f) for f in fields}, {f: r.get(g) for f, g in fields.items()})
    return pairs


def slope_pairs(new: dict, ref: dict):
    c = new["c"]
    names = ["reach", "grasp", "transport", "place"]
    pairs = {"n_landmarks": (c["n_landmarks"], ref["n_landmarks"]),
             "n_per_phase": (c["n_per_phase"], ref["n_per_phase"])}
    for cond in ("natural", "frozen"):
        pairs[f"decoded correctly, {cond}"] = (
            {k: round(c["decoded_correctly_pct"][cond][k], 2) for k in names},
            {k: ref["decoded_correctly_pct"][cond][k] for k in names})
    for k in names:
        n, r = c["frozen_most_read"][k], ref["frozen_destination"][k]
        pairs[f"frozen {k}"] = (
            {"top": n["top"], "top_pct": round(n["top_pct"], 2), "second": n["second"],
             "second_pct": round(n["second_pct"], 2)},
            {f: r[f] for f in ("top", "top_pct", "second", "second_pct")})
    return pairs


KIND_OF_PHASE = {"reach": ("reach",), "grasp": ("grasp_open", "grasp_closed"),
                 "transport": ("transport",), "place": ("place_closed",)}


def counts_pairs(new: dict, ref: dict):
    d = new["d"]
    pairs = {"n_episodes": (d["n_episodes"], ref["n_files"]),
             "n_landmarks": (d["n_landmarks"], ref["n_landmarks"])}
    for k, kinds in KIND_OF_PHASE.items():
        pairs[f"n {k}"] = (d["per_phase"][k]["n"],
                           sum(ref["per_kind"][kd]["n"] for kd in kinds if kd in ref["per_kind"]))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("what", choices=("scan", "decode", "gripper", "slope", "counts"))
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    a = ap.parse_args(argv)
    new, ref = json.loads(a.new.read_text()), json.loads(a.ref.read_text())
    pairs = {"scan": scan_pairs, "decode": decode_pairs, "gripper": gripper_pairs,
             "slope": slope_pairs, "counts": counts_pairs}[a.what](new, ref)
    print(f"{a.what}\n  new {a.new}\n  ref {a.ref}")
    ok = True
    for name, (n, r) in pairs.items():
        diffs = _walk(n, r)
        if diffs:
            ok = False
            print(f"  DIFF  {name}")
            for p, x, y in diffs[:10]:
                print(f"        {p}: {x!r} vs {y!r}")
        else:
            print(f"  equal {name}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
