"""
test_region_weight.py -- smoke/characterization tests for the region_weight logic in
per_neuron_simple_angelov_nn_test.py.

The loss functions there are nested closures (can't be imported directly), so this file
does two things:

  1. SOURCE GUARD (test_source_matches): greps the production file for the exact
     expressions this test mirrors. If someone edits the formula, these tests stop
     being a faithful mirror -> this guard fails loudly, forcing the test to be updated.

  2. PROPERTY TESTS: reproduce the exact expressions (region box weight L776, weighted
     Ids L787, weighted gm _mse L805, nmse norms L734/739) on synthetic tensors and a
     tiny autograd model, and assert the three claims:
        - only the specified box is up-weighted (inside = 1+w, outside = 1)
        - the weight multiplies BOTH the Ids loss and the gm losses
        - nmse normalization is preserved, and everything reduces to plain MSE when off

Run:  python test_region_weight.py
"""
import os, re, sys
import torch
import torch.nn as nn

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "per_neuron_simple_angelov_nn_test.py")
torch.manual_seed(0)
_eps = 1e-12
crit = nn.MSELoss()

def approx(a, b, tol=1e-9): return abs(float(a) - float(b)) <= tol * (1 + abs(float(b)))


# ---- exact reproductions of the production expressions --------------------------
def region_box_weight(Vgs, Vds, weight, vgs_lo, vgs_hi, vds_lo, vds_hi):
    """Mirror of L771-776."""
    inbox = torch.ones_like(Vgs, dtype=torch.bool)
    if vgs_lo is not None: inbox &= (Vgs >= vgs_lo)
    if vgs_hi is not None: inbox &= (Vgs <= vgs_hi)
    if vds_lo is not None: inbox &= (Vds >= vds_lo)
    if vds_hi is not None: inbox &= (Vds <= vds_hi)
    return 1.0 + weight * inbox.to(Vgs.dtype), inbox

def weighted_nmse(pred, target, w, norm):
    """Mirror of the Ids term L787 / gm _mse L805 (weighted mean), then /norm."""
    if w is None:
        return crit(pred, target) / norm
    return (w * (pred - target) ** 2).sum() / w.sum() / norm


# ---------------------------------------------------------------------------------
def test_source_matches():
    src = open(SRC, encoding="utf-8").read()
    needles = [
        "region_w = 1.0 + args.region_weight * _inbox.to(X_train_t.dtype)",   # L776
        "(w * (preds - y_train_t) ** 2).sum() / w.sum() / ids_norm",          # L787
        "(_rw * (p - t) ** 2).sum() / _rw.sum() if _rw is not None else criterion(p, t)",  # L805
        "_rw = region_w[gm_loss_mask] if gm_loss_mask is not None else region_w",           # L803
    ]
    for n in needles:
        assert n in src, f"SOURCE DRIFT: expected expression not found -> {n!r}"
    print("  [ok] source guard: production formulas match what this test mirrors")


def test_box_only():
    # grid of Vgs in [-4,2], Vds in [-2,18]
    g = torch.linspace(-4, 2, 13)
    d = torch.linspace(-2, 18, 11)
    Vgs, Vds = torch.meshgrid(g, d, indexing="ij")
    Vgs, Vds = Vgs.reshape(-1), Vds.reshape(-1)
    w, inbox = region_box_weight(Vgs, Vds, weight=4.0, vgs_lo=-3, vgs_hi=0, vds_lo=0, vds_hi=15)
    # inside -> 1+4=5, outside -> 1
    assert torch.all(w[inbox] == 5.0), "in-box weight must be 1+weight"
    assert torch.all(w[~inbox] == 1.0), "out-of-box weight must be exactly 1"
    # mask really is the box and nothing else
    manual = (Vgs >= -3) & (Vgs <= 0) & (Vds >= 0) & (Vds <= 15)
    assert torch.equal(inbox, manual), "mask must equal exactly the specified box"
    # unbounded side: leave vds_hi=None -> box extends to +inf in Vds
    w2, inbox2 = region_box_weight(Vgs, Vds, 4.0, -3, 0, 0, None)
    assert inbox2.sum() > inbox.sum(), "dropping a bound must not shrink the box"
    print(f"  [ok] box-only: {int(inbox.sum())}/{len(Vgs)} pts weighted 5x, rest 1x; unbounded side widens box")


def test_reduces_to_plain_mse_when_off():
    pred = torch.randn(200); tgt = torch.randn(200); norm = 3.7
    off = weighted_nmse(pred, tgt, None, norm)                 # region_weight=0 path
    w_ones = torch.ones_like(pred)                             # weight=0 -> region_w all 1
    weighted = weighted_nmse(pred, tgt, w_ones, norm)
    assert approx(off, weighted), "weight=0 must equal plain criterion/norm"
    print(f"  [ok] reduces to plain nmse when off: {float(off):.6f} == {float(weighted):.6f}")


def test_inbox_error_is_upweighted():
    # residual ONLY inside the box; higher weight => higher loss (box emphasized)
    n = 400
    inbox = torch.zeros(n, dtype=torch.bool); inbox[:100] = True
    err = torch.zeros(n); err[:100] = 0.5                      # error only in box
    pred = err; tgt = torch.zeros(n)
    losses = []
    for wt in (0.0, 1.0, 4.0):
        w = 1.0 + wt * inbox.to(torch.float32)
        losses.append(float(weighted_nmse(pred, tgt, w, 1.0)))
    assert losses[0] < losses[1] < losses[2], f"in-box error must weigh more as weight grows: {losses}"
    print(f"  [ok] in-box error up-weighted: loss(w=0,1,4) = {[round(x,5) for x in losses]}")


def test_normalization_scales():
    pred = torch.randn(150); tgt = torch.randn(150); w = 1.0 + 3.0 * (torch.rand(150) > .5).float()
    l1 = weighted_nmse(pred, tgt, w, 1.0)
    l2 = weighted_nmse(pred, tgt, w, 5.0)
    assert approx(l1 / 5.0, l2), "dividing by norm=5 must scale loss by 1/5 (nmse preserved)"
    print(f"  [ok] normalization applied: loss/5 == loss(norm=5)  ({float(l1)/5:.6f})")


def test_gm_losses_are_weighted():
    # tiny model f: (Vgs,Vds)->Ids ; gm1 = d Ids/d Vgs via autograd (mirrors L806-812)
    net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    X = torch.rand(120, 2, requires_grad=True) * torch.tensor([6.0, 20.0]) + torch.tensor([-4.0, -2.0])
    preds = net(X).squeeze(-1)
    gm1 = torch.autograd.grad(preds, X, torch.ones_like(preds), create_graph=True)[0][:, 0]
    tgt = torch.zeros_like(gm1)
    Vgs, Vds = X[:, 0].detach(), X[:, 1].detach()
    region_w, inbox = region_box_weight(Vgs, Vds, 4.0, -3, 0, 0, 15)
    gm_norm = gm1.detach().pow(2).mean().item() + _eps

    def _mse(p, t, rw):   # mirror of L805
        return (rw * (p - t) ** 2).sum() / rw.sum() if rw is not None else crit(p, t)

    off = _mse(gm1, tgt, None) / gm_norm
    on  = _mse(gm1, tgt, region_w) / gm_norm
    ones = _mse(gm1, tgt, torch.ones_like(gm1)) / gm_norm
    assert approx(off, ones), "gm: weight=0 (all-ones) must equal plain criterion"
    assert not approx(off, on, tol=1e-6), "gm: a real box weight must change the gm loss (proves gm IS weighted)"
    # and it must be backprop-able (the whole point)
    on.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), "gm loss must backprop"
    print(f"  [ok] gm losses weighted+differentiable: off={float(off):.6f} weighted={float(on):.6f} (differ), grads finite")


def main():
    tests = [test_source_matches, test_box_only, test_reduces_to_plain_mse_when_off,
             test_inbox_error_is_upweighted, test_normalization_scales, test_gm_losses_are_weighted]
    print("region_weight logic — smoke tests\n" + "-" * 50)
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1; print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print("-" * 50)
    print(f"{'ALL PASSED' if not failed else str(failed)+' FAILED'}  ({len(tests)-failed}/{len(tests)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
