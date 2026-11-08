"""Shared constrained curve evaluation for real and simulated states."""
import numpy as np
import torch


def curve(tau, money, xi, rep, scale):
    """Return normalized call, strike slope and normalized-factor derivatives."""
    times = np.unique(rep["grid_t"])
    strikes = np.unique(rep["grid_k"])
    shape = tau.shape
    tau, money, xi = tau.ravel(), money.ravel(), xi.reshape(-1, 2)
    centers = rep["center"].reshape(len(times), len(strikes))
    basis = (rep["basis"] * scale[:, None]).reshape(2, len(times), len(strikes))
    n = len(tau)
    t1 = np.minimum(np.searchsorted(times, tau), len(times)-1)
    t0 = t1-1
    a = np.clip((tau-np.r_[0, times][t1])/(times[t1]-np.r_[0, times][t1]), 0, 1)
    j = np.clip(np.searchsorted(strikes, money, side="right")-1, 0, len(strikes)-2)
    k0, k1 = strikes[j], strikes[j+1]
    payoff = np.maximum(1-money, 0)
    zero = np.zeros((n, 2))

    def at(t):
        idx = np.maximum(t, 0)
        p = centers[idx, j] + np.sum(xi*basis[:, idx, j].T, axis=1)
        b = basis[:, idx, j].T
        d = basis[:, idx, j+1].T-b
        q = centers[idx, j+1] - centers[idx, j] + np.sum(xi*d, axis=1)
        slope = q/(k1-k0)
        d = d/(k1-k0)[:, None]
        price = p+slope*(money-k0)
        deriv = b+d*(money-k0)[:, None]
        left = money < strikes[0]
        price[left] = 1+(p[left]-1)*money[left]/strikes[0]
        slope[left] = (p[left]-1)/strikes[0]
        deriv[left] = b[left]*money[left, None]/strikes[0]
        right = money > strikes[-1]
        constrained = right & ((slope < -1) | (slope > 0))
        price[constrained] = (p+q)[constrained] + np.clip(slope[constrained], -1, 0)*(money[constrained]-k1[constrained])
        deriv[constrained] = (b+d*(k1-k0)[:, None])[constrained]
        slope[constrained] = np.clip(slope[constrained], -1, 0)
        floor = price < payoff
        cap = price > 1
        intrinsic = floor | (t < 0)
        price[intrinsic], slope[intrinsic], deriv[intrinsic] = payoff[intrinsic], -(money[intrinsic] < 1).astype(float), 0
        cap &= ~intrinsic
        price[cap], slope[cap], deriv[cap] = 1, 0, 0
        return price, slope, deriv

    p0, s0, d0 = at(t0)
    p1, s1, d1 = at(t1)
    return ((p0+(p1-p0)*a).reshape(shape), (s0+(s1-s0)*a).reshape(shape),
            (d0+(d1-d0)*a[:, None]).reshape(*shape, 2))


def bs_delta(spot, strike, quote, tau):
    """Zero-rate, zero-yield implied BS delta; bad quotes carry a false mask."""
    spot, strike, quote, tau = (torch.as_tensor(x, dtype=torch.float64)
                                for x in (spot, strike, quote, tau))
    k, c = strike/spot, quote/spot
    valid = (tau > 0) & (k > 0) & (quote > (spot-strike).clamp(min=0)) & (quote < spot) & torch.isfinite(c)
    result = torch.zeros_like(c, dtype=torch.float32)
    if valid.any():
        m, c0 = -k[valid].log(), c[valid]
        k0 = k[valid]
        lo, hi = torch.zeros_like(c0), torch.full_like(c0, 12.)
        for _ in range(32):
            vol = (lo+hi)/2
            d1 = m/vol+vol/2
            call = torch.special.ndtr(d1)-k0*torch.special.ndtr(d1-vol)
            lo = torch.where(call < c0, vol, lo)
            hi = torch.where(call < c0, hi, vol)
        vol = (lo+hi)/2
        d1 = m/vol+vol/2
        d = torch.special.ndtr(d1)
        good = torch.isfinite(d) & ((d-k0*torch.special.ndtr(d1-vol)-c0).abs() <= 1e-7)
        result[valid] = d.to(torch.float32)
        valid[valid.clone()] = good
    return result, valid


def reconstructed_delta(spot, strike, tau, xi, rep, scale):
    outputs, masks = [], []
    strike = np.broadcast_to(strike, spot.shape)
    for sl in np.array_split(np.arange(len(spot)), max(1, (len(spot)+511)//512)):
        s, k, t = spot[sl], strike[sl], tau[sl]
        c, _, _ = curve(t, k/s, xi[sl], rep, scale)
        value, ok = bs_delta(s, k, c*s, t)
        # Current intrinsic delta is the finite fallback; same in both branches.
        outputs.append(torch.where(ok, value, torch.from_numpy((s > k).astype("f"))))
        masks.append(ok)
    return torch.cat(outputs), torch.cat(masks)

