"""Gripper edit optimizer.

An edit is a vector delta (768,) added to the output of one action-expert block on the final flow
step only. `optimize_delta` fits delta with Adam so that the decoded gripper command moves to
closed (+1) or open (-1) while the decoded arm actions stay at a target. Loss per iteration, over
the H = 16 actions of the predicted chunk:

    lam * ||delta||^2
  + grip_weight   * mean((gripper - g*)^2)            g* = +1 (close) or -1 (open)
  + arm_weight    * mean((arm - arm_target)^2)
  + smooth_weight * mean((arm[t+1] - arm[t])^2)

The lowest-loss delta seen during the iterations is returned. The flow prefix (steps 0..N-2) does
not depend on delta, so the caller computes it once and passes `final_step_fn(delta)`, which runs
only the final step (see `vlageom.sim.engine.BranchEngine.fit_deltas`).

`optimize_delta_basis` fits the same loss with the edit restricted to delta = mean + V[:, :k] a
(Adam on the coefficients a; `BranchEngine.fit_deltas_basis`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRIP = 7  # action layout: arm joints [0:7] + gripper [7]


@dataclass
class DeltaFit:
    delta: np.ndarray        # (d,) the fitted edit
    grip_before: float       # mean decoded gripper over the chunk without the edit
    grip_after: float        # mean decoded gripper over the chunk with the edit
    arm_leakage: float       # RMS(arm_after - arm_target)
    norm: float              # ||delta||
    arm_vs_natural: float = float("nan")  # RMS(arm_after - arm without the edit)
    arm_jerk: float = float("nan")        # RMS step-to-step arm change, with the edit
    nat_jerk: float = float("nan")        # RMS step-to-step arm change, without the edit
    arm_span: float = float("nan")        # mean over joints of (max - min) over the H actions
    # the same metrics over the executed actions (the first min(8, H)):
    grip_exec: float = float("nan")
    arm_leak_exec: float = float("nan")
    arm_jerk_exec: float = float("nan")
    arm_span_exec: float = float("nan")
    nat_jerk_exec: float = float("nan")


def optimize_delta(
    final_step_fn,
    new_delta_fn,
    *,
    target_sign: float,
    iters: int = 120,
    lr: float = 0.1,
    lam: float = 1e-4,
    grip_open_val: float = -1.0,
    grip_closed_val: float = 1.0,
    n_target_steps: int | None = None,
    arm_target: np.ndarray | None = None,
    arm_weight: float = 1.0,
    grip_weight: float = 1.0,
    smooth_weight: float = 0.0,
) -> DeltaFit:
    """Fit delta so the decoded gripper goes to closed (target_sign > 0) or open (< 0).

    final_step_fn : callable(delta or None) -> torch tensor (B, H, 8), the action chunk after the
        final flow step with delta added at the fixed layer (None = no edit); differentiable in delta.
    new_delta_fn : callable() -> zero leaf tensor (d,) with requires_grad.
    n_target_steps : the gripper term uses only the first n actions (None = all H).
    arm_target : normalized arm action target, (GRIP,) broadcast over H or (H, GRIP). None = the
        arm actions predicted without the edit. `BranchEngine.freeze_arm_target(qpos)` gives the
        target that keeps the arm at its current joint angles.
    arm_weight : scalar, or per-joint (GRIP,) weights.
    smooth_weight : weight of the step-to-step arm smoothness term (0 = off).
    """
    import torch

    with torch.no_grad():
        a0 = final_step_fn(None)
    arm0 = a0[0, :, :GRIP].float()
    grip_tgt = torch.tensor(grip_closed_val if target_sign > 0 else grip_open_val,
                            device=arm0.device, dtype=torch.float32)

    if arm_target is None:
        arm_tgt = arm0
    else:
        at = torch.as_tensor(np.asarray(arm_target, dtype=np.float32),
                             device=arm0.device, dtype=torch.float32)
        arm_tgt = at.unsqueeze(0).expand_as(arm0) if at.ndim == 1 else at

    aw = torch.as_tensor(np.asarray(arm_weight, dtype=np.float32), device=arm0.device)

    delta = new_delta_fn()
    opt = torch.optim.Adam([delta], lr=lr)
    best = (float("inf"), None)
    for _ in range(iters):
        opt.zero_grad()
        aout = final_step_fn(delta)
        grip_v = aout[0, :, GRIP].float()
        gv = grip_v if n_target_steps is None else grip_v[:n_target_steps]
        arm = aout[0, :, :GRIP].float()
        arm_dev = (arm - arm_tgt) ** 2
        arm_loss = (arm_dev * aw).mean() if aw.ndim > 0 else aw * arm_dev.mean()
        loss = (lam * (delta ** 2).sum()
                + grip_weight * ((gv - grip_tgt) ** 2).mean()
                + arm_loss)
        if smooth_weight > 0:
            loss = loss + smooth_weight * ((arm[1:] - arm[:-1]) ** 2).mean()
        lv = float(loss.detach())
        if lv < best[0]:
            best = (lv, delta.detach().clone())
        loss.backward()
        opt.step()

    with torch.no_grad():
        sf = final_step_fn(best[1])
    d = best[1].detach().float().cpu().numpy()
    return _measure(a0, sf, arm_tgt, d)


def optimize_delta_basis(final_step_fn, *, basis, mean, n_coeffs, target_sign, init_coeffs=None,
                         iters=1000, lr=0.1, lam=1e-4, arm_target=None, arm_weight=1.0,
                         grip_weight=1.0, smooth_weight=0.0):
    """`vlageom.steer.delta.optimize_delta` with the edit restricted to
    delta = mean + basis[:, :n_coeffs] @ a: Adam on the coefficients a (n_coeffs,), same loss
    (the norm term on the full delta), lowest-loss a kept.

    basis (d, K) and mean (d,): numpy or torch. init_coeffs: the start value of a (its first
    n_coeffs entries; default 0). n_coeffs <= 0 evaluates delta = mean without optimization.
    Returns (DeltaFit, a as numpy (n_coeffs,), {"k_eff": n_coeffs, "loss": lowest loss})."""
    import torch

    with torch.no_grad():
        a0 = final_step_fn(None)
    arm0 = a0[0, :, :GRIP].float()
    dev = arm0.device
    grip_tgt = torch.tensor(1.0 if target_sign > 0 else -1.0, device=dev, dtype=torch.float32)
    if arm_target is None:
        arm_tgt = arm0
    else:
        at = torch.as_tensor(np.asarray(arm_target, dtype=np.float32), device=dev,
                             dtype=torch.float32)
        arm_tgt = at.unsqueeze(0).expand_as(arm0) if at.ndim == 1 else at
    aw = torch.as_tensor(np.asarray(arm_weight, dtype=np.float32), device=dev)

    def _t(x):
        if torch.is_tensor(x):
            return x.to(device=dev, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x, np.float32), device=dev)

    basis_t = _t(basis)
    d = int(basis_t.shape[0])
    mean_t = None if mean is None else _t(mean)

    if n_coeffs is None or int(n_coeffs) <= 0:
        with torch.no_grad():
            sf = final_step_fn(mean_t if mean_t is not None else None)
        delta_np = mean_t.detach().cpu().numpy() if mean_t is not None else np.zeros(d, np.float32)
        return (_measure(a0, sf, arm_tgt, delta_np), np.zeros(0, np.float32),
                {"k_eff": 0, "loss": float("nan")})

    k = int(n_coeffs)
    V = basis_t[:, :k].contiguous()
    a_init = np.zeros(k, np.float32)
    if init_coeffs is not None:
        ic = (init_coeffs.detach().cpu().numpy() if torch.is_tensor(init_coeffs)
              else np.asarray(init_coeffs, np.float32)).reshape(-1)
        m = min(k, ic.shape[0])
        a_init[:m] = ic[:m]
    a = torch.tensor(a_init, device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    best = (float("inf"), None)
    for _ in range(iters):
        opt.zero_grad()
        delta = (V @ a) if mean_t is None else (mean_t + V @ a)
        aout = final_step_fn(delta)
        gv = aout[0, :, GRIP].float()
        arm = aout[0, :, :GRIP].float()
        arm_dev = (arm - arm_tgt) ** 2
        arm_loss = (arm_dev * aw).mean() if aw.ndim > 0 else aw * arm_dev.mean()
        loss = (lam * (delta ** 2).sum()
                + grip_weight * ((gv - grip_tgt) ** 2).mean()
                + arm_loss)
        if smooth_weight > 0:
            loss = loss + smooth_weight * ((arm[1:] - arm[:-1]) ** 2).mean()
        lv = float(loss.detach())
        if lv < best[0]:
            best = (lv, a.detach().clone())
        loss.backward()
        opt.step()

    with torch.no_grad():
        delta_best = (V @ best[1]) if mean_t is None else (mean_t + V @ best[1])
        sf = final_step_fn(delta_best)
    fit = _measure(a0, sf, arm_tgt, delta_best.detach().cpu().numpy())
    return fit, best[1].detach().cpu().numpy(), {"k_eff": int(V.shape[1]), "loss": float(best[0])}


def _measure(a0, sf, arm_tgt, delta_np) -> DeltaFit:
    """DeltaFit metrics from the chunk without (`a0`) and with (`sf`) the edit, both (B, H, 8)
    torch tensors, the arm target `arm_tgt` (H, GRIP) and the edit `delta_np` (d,)."""
    arm0 = a0[0, :, :GRIP].float()
    grip_before = float(a0[0, :, GRIP].float().mean())
    grip_after = float(sf[0, :, GRIP].float().mean())
    arm_after = sf[0, :, :GRIP].float()
    arm_leakage = float(((arm_after - arm_tgt) ** 2).mean().sqrt())
    arm_vs_natural = float(((arm_after - arm0) ** 2).mean().sqrt())
    arm_jerk = float(((arm_after[1:] - arm_after[:-1]) ** 2).mean().sqrt())
    nat_jerk = float(((arm0[1:] - arm0[:-1]) ** 2).mean().sqrt())
    arm_span = float((arm_after.max(0).values - arm_after.min(0).values).mean())
    ne = min(8, arm_after.shape[0])
    arm_e = arm_after[:ne]
    grip_exec = float(sf[0, :ne, GRIP].float().mean())
    arm_leak_exec = float(((arm_e - arm_tgt[:ne]) ** 2).mean().sqrt())
    arm_jerk_exec = float(((arm_e[1:] - arm_e[:-1]) ** 2).mean().sqrt()) if ne > 1 else 0.0
    arm_span_exec = float((arm_e.max(0).values - arm_e.min(0).values).mean())
    arm0_e = arm0[:ne]
    nat_jerk_exec = float(((arm0_e[1:] - arm0_e[:-1]) ** 2).mean().sqrt()) if ne > 1 else 0.0
    d = np.asarray(delta_np, dtype=np.float32).reshape(-1)
    return DeltaFit(delta=d, grip_before=grip_before, grip_after=grip_after,
                    arm_leakage=arm_leakage, norm=float(np.linalg.norm(d)),
                    arm_vs_natural=arm_vs_natural, arm_jerk=arm_jerk, nat_jerk=nat_jerk,
                    arm_span=arm_span,
                    grip_exec=grip_exec, arm_leak_exec=arm_leak_exec,
                    arm_jerk_exec=arm_jerk_exec, arm_span_exec=arm_span_exec,
                    nat_jerk_exec=nat_jerk_exec)
