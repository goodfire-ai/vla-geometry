"""Read activations from the policy for an observation or a captured query context.

  encode_obs     run the policy on one observation (one model query, flow noise seeded) and return
                 the action expert's activation at the final flow step for each requested layer
  run_chunks     live rollout of the first `n_chunks` model queries from an observation, capturing
                 each query's generate_actions kwargs, predicted chunk and the object / EE state
  encode_query   for a captured context: VLM backbone activations pooled over 11 token groups
                 (`VARIANTS`) at `llm_layers`, and per-token action-expert activations at
                 `ae_layers`

Token layout of the VLM input (MolmoBot-DROID): [BOS] + per image block 6 header tokens
"Image {N} View {M}" + <im_start> + 196 x <im_patch> + <im_end>, then the text tail
"<|im_start|>user\\ndemo: {task}<|im_end|>\\n<|im_start|>assistant\\n". The first query has 2 image
blocks (exo, wrist); later queries have 4 (exo and wrist 8 steps earlier, then the current exo and
wrist). The last two patch runs are always the current exo and wrist frames.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Layers and token groups stored by the object-teleport harvests.
LLM_LAYERS = [0, 4, 8, 12, 16, 20, 21, 24, 27, 29, 31, 33, 35]
AE_LAYERS = [0, 4, 8, 9, 12, 16, 20, 24, 28, 32, 35]
VARIANTS = ["seq_all", "img_all", "img_exo", "img_wrist", "img_prev", "hdr_txt",
            "txt_tail", "txt_instr", "phrase_pickup", "phrase_pickup_end", "last_tok"]
OPTIONAL_VARIANTS = {"img_prev", "phrase_pickup", "phrase_pickup_end"}   # may be None for a query

# ---- token layout ----------------------------------------------------------------------------- #
IM_START, IM_END, IM_PATCH = 151936, 151937, 151938
BOS = 151645            # Qwen <|im_end|>, used as BOS and as the user-turn terminator
CHAT_START = 151644     # Qwen <|im_start|> (chat template)
SPECIAL_MIN = IM_START  # ids >= this are multimodal special tokens


def patch_runs(ids):
    """Consecutive runs of <im_patch> in ids -> [(start, length), ...]."""
    mask = np.asarray(ids) == IM_PATCH
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def token_map(ids) -> dict:
    """Parse one input_ids row -> dict(runs, n_img, cur_exo, cur_wrist, tail_start, seq_len).
    cur_exo / cur_wrist are the (start, len) of the last two patch runs; tail_start is the index
    after the last <im_end>."""
    ids = np.asarray(ids)
    runs = patch_runs(ids)
    if len(runs) < 2:
        raise ValueError(f"expected >=2 image-patch runs, got {len(runs)}")
    ends = np.where(ids == IM_END)[0]
    tail_start = int(ends[-1]) + 1 if len(ends) else int(runs[-1][0] + runs[-1][1])
    return {"runs": runs, "n_img": len(runs),
            "cur_exo": runs[-2], "cur_wrist": runs[-1],
            "tail_start": tail_start, "seq_len": int(len(ids))}


def find_subseq(hay, needle) -> int:
    """First start index of `needle` (list of ids) in `hay` (1-D), or -1."""
    hay = np.asarray(hay)
    needle = list(needle)
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if all(int(hay[i + j]) == int(needle[j]) for j in range(len(needle))):
            return i
    return -1


def noun_span_from_ids(tail_ids, name_ids):
    """(start, end, matched): span of `name_ids` in the text tail; if not found, the template span
    from token 8 ("<|im_start|>user\\ndemo: Pick up the" is 8 tokens) to the first BOS."""
    tail_ids = np.asarray(tail_ids)
    s = find_subseq(tail_ids, name_ids)
    if s >= 0:
        return int(s), int(s + len(name_ids)), True
    end = np.where(tail_ids == BOS)[0]
    e = int(end[0]) if len(end) else len(tail_ids)
    return 8, e, False


def tail_special_mask(tail_ids):
    """True for control tokens of the text tail (multimodal specials, <|im_start|>, <|im_end|>)."""
    tail_ids = np.asarray(tail_ids)
    return (tail_ids >= SPECIAL_MIN) | (tail_ids == BOS) | (tail_ids == CHAT_START)


def llm_variant_pools(acts_seq, ids) -> dict:
    """Pool one layer's VLM activations (seq_len, d) over the token groups in `VARIANTS`:
    seq_all (all tokens), img_all (all patches), img_exo / img_wrist (current-frame patches),
    img_prev (earlier-frame patches; None on the first query), hdr_txt (non-patch tokens before the
    text tail), txt_tail (the text tail), txt_instr (non-special tail tokens), phrase_pickup (the
    object phrase), phrase_pickup_end (its last token), last_tok (the final token)."""
    tm = token_map(ids)
    n = len(ids)
    pos = np.arange(n)
    img = np.zeros(n, bool)
    for (s, ln) in tm["runs"]:
        img[s:s + ln] = True
    exo_s, exo_l = tm["cur_exo"]
    wr_s, wr_l = tm["cur_wrist"]
    tail = pos >= tm["tail_start"]
    hdr = ~img & ~tail
    tail_ids = ids[tm["tail_start"]:]
    nonspec = ~tail_special_mask(tail_ids)
    ps, pe, _matched = noun_span_from_ids(tail_ids, [])
    phrase = np.zeros(n, bool)
    phrase[tm["tail_start"] + ps: tm["tail_start"] + pe] = True

    def mp(mask):
        return acts_seq[mask].mean(0) if mask.any() else None

    out = {"seq_all": acts_seq.mean(0), "img_all": mp(img),
           "img_exo": mp((pos >= exo_s) & (pos < exo_s + exo_l)),
           "img_wrist": mp((pos >= wr_s) & (pos < wr_s + wr_l)),
           "hdr_txt": mp(hdr), "txt_tail": mp(tail), "last_tok": acts_seq[-1]}
    prev = img.copy()
    prev[exo_s:exo_s + exo_l] = False
    prev[wr_s:wr_s + wr_l] = False
    out["img_prev"] = mp(prev) if tm["n_img"] > 2 else None
    tn = np.zeros(n, bool)
    tn[tm["tail_start"]:] = nonspec
    out["txt_instr"] = mp(tn)
    out["phrase_pickup"] = mp(phrase)
    out["phrase_pickup_end"] = acts_seq[tm["tail_start"] + pe - 1] if pe > ps else None
    return out


def capture_kw(eng, obs, seed):
    """`policy.get_action(obs)` with generate_actions wrapped so that the flow noise is seeded by
    `seed`; returns the generate_actions kwargs of that query (None if no query fired)."""
    import torch
    cap = {}

    def gen(**kw):
        kw2, steps = eng._seeded_kw(kw, seed)
        cap["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                     for k, v in kw2.items() if k != "generator"}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return eng._gen_orig(num_steps=steps, **kw2)

    orig = eng.model.generate_actions
    eng.model.generate_actions = gen
    try:
        eng.policy.get_action(obs)
    finally:
        eng.model.generate_actions = orig
    return cap.get("kw")


def capture_blocks(eng, kw, layers):
    """Run the VLM backbone on captured generate_actions kwargs with forward hooks on
    transformer.blocks.{L}. Returns ({L: (seq_len, 2560) float32}, input_ids) or (None, None) when
    a hook did not fire."""
    cap: dict = {}
    handles = []

    def mk(L):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            cap[L] = t.detach()[0].float().cpu().numpy()
        return hook

    for L in layers:
        handles.append(eng.model.get_submodule(f"transformer.blocks.{L}").register_forward_hook(mk(L)))
    try:
        eng._build_backbone_cache(kw)
    finally:
        for h in handles:
            h.remove()
    if len(cap) != len(layers):
        return None, None
    return cap, kw["input_ids"][0].detach().cpu().numpy()


# ---- action-expert readout -------------------------------------------------------------------- #
def encode_obs(eng, obs, layers, seed, per_token=False):
    """Run the policy on `obs` (one model query with flow noise seeded by `seed`) and return
    {layer: activation} of action_expert.blocks.{layer} at the final flow step: (768,) mean over the
    action tokens, or (n_token, 768) with per_token=True. None if `obs` did not trigger a query
    (the caller restores a pre-query snapshot first)."""
    kw = capture_kw(eng, obs, seed)
    if kw is None:
        return None
    cache = eng._build_backbone_cache(kw)
    prefix = eng._compute_prefix(cache, seed)
    out = {}
    for L in layers:
        _, act = eng._final_step(cache, prefix, None, L, grad=False, per_token=per_token)
        out[L] = act.detach().float().cpu().numpy()
    return out


def run_chunks(eng, obs, n_chunks, max_loop):
    """Roll the policy forward from `obs` until `n_chunks` model queries have fired (at most
    `max_loop` env steps), without executing past the last query.

    Query q uses flow noise seeded with `eng.scene_seed + 1000 * q`. Returns (caps, states):
    caps[q] = {"kw": generate_actions kwargs, "actions": predicted chunk (H, 8), "q": q};
    states[q] = {"obj_w": object world position, "ee_b": EE base-frame position, "ee44": EE pose,
    "step": env step}, read right after the query fires (the state the model saw). Stops early
    if an object or EE read fails."""
    import torch
    caps, states = [], []

    def gen(**kw):
        q = eng._chunk
        eng._chunk += 1
        seed = eng.scene_seed + 1000 * q
        kw2, steps = eng._seeded_kw(kw, seed)
        kwc = {k: (v.detach().clone() if torch.is_tensor(v) else v)
               for k, v in kw2.items() if k != "generator"}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eng._gen_orig(num_steps=steps, **kw2)
        caps.append({"kw": kwc, "actions": out[0].detach().float().cpu().numpy(), "q": q})
        return out

    orig = eng.model.generate_actions
    eng.model.generate_actions = gen
    try:
        step = 0
        while step < max_loop and not eng.task.is_done():
            pre = len(caps)
            action = eng.policy.get_action(obs)
            if action is None:
                break
            if len(caps) > pre:
                ow = np.asarray(eng.read_object(), np.float64)
                ee = np.asarray(eng.read_ee(), np.float64)
                if ow.size != 3 or ee.size != 3:
                    break
                states.append({"obj_w": ow, "ee_b": ee,
                               "ee44": eng.read_ee_full(), "step": step})
            if len(caps) >= n_chunks:
                break
            obs, *_ = eng.task.step(action)
            step += 1
    finally:
        eng.model.generate_actions = orig
    return caps, states


def encode_query(eng, kw, llm_layers, ae_layers, seed):
    """One captured query context -> (llm, ae_tok, ids):
    llm = {variant: (len(llm_layers), 2560) float32 or None} (see `llm_variant_pools`; None when a
    variant is empty at some layer), ae_tok = (len(ae_layers), n_token, 768) float32 action-expert
    activations at the final flow step (flow noise seeded by `seed`), ids = input_ids (seq_len,).
    Returns None if a backbone hook did not fire or token pooling failed."""
    seq_cap: dict = {}
    handles = []

    def mk(L):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            seq_cap[L] = t.detach()[0].float().cpu().numpy()
        return hook

    for L in llm_layers:
        handles.append(eng.model.get_submodule(f"transformer.blocks.{L}").register_forward_hook(mk(L)))
    try:
        cache = eng._build_backbone_cache(kw)
    finally:
        for h in handles:
            h.remove()
    if len(seq_cap) != len(llm_layers):
        return None
    ids = kw["input_ids"][0].detach().cpu().numpy()

    llm: dict = {}
    for L in llm_layers:
        try:
            pools = llm_variant_pools(seq_cap[L], ids)
        except Exception as exc:
            log.warning("variant pooling failed: %s", exc)
            return None
        for v in VARIANTS:
            llm.setdefault(v, []).append(pools[v])
    for v in VARIANTS:
        llm[v] = (np.stack(llm[v]).astype(np.float32)
                  if all(x is not None for x in llm[v]) else None)

    prefix = eng._compute_prefix(cache, seed)
    toks = []
    for L in ae_layers:
        _, act = eng._final_step(cache, prefix, None, L, grad=False, per_token=True)
        toks.append(act.detach().float().cpu().numpy())
    return llm, np.stack(toks).astype(np.float32), ids
