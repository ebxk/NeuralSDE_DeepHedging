"""Zero-rate, zero-dividend normalized call surfaces (S=F=1).

Public k is strike/spot, never log strike. Diagnostics certify a finite grid,
not a global arbitrage theorem. No input files are read or written.
"""
import time
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import ndtr


def _call_w(k, w):
    k, w = np.broadcast_arrays(np.asarray(k, float), np.asarray(w, float))
    root = np.sqrt(np.maximum(w, 1e-30))
    d = -np.log(k) / root + root / 2
    return np.where(w > 0, ndtr(d) - k * ndtr(d-root), np.maximum(1-k, 0))


def bs_call(t, k, sigma):
    """Normalized BS call; nonpositive time returns intrinsic value."""
    t, k, sigma = np.broadcast_arrays(np.asarray(t, float), np.asarray(k, float), np.asarray(sigma, float))
    if np.any(k <= 0) or np.any(sigma < 0):
        raise ValueError('positive strikes and nonnegative volatility required')
    return _call_w(k, np.maximum(t, 0)*sigma*sigma)


def implied_vol(t, k, price):
    """BS inversion; invalid/unidentifiable observations return NaN.

    Intrinsic at positive time maps to zero; upper bound 1 has infinite IV.
    Finite IV search is capped at 10, a declared financial bound.
    """
    t, k, price = np.broadcast_arrays(np.asarray(t, float), np.asarray(k, float), np.asarray(price, float))
    result = np.full(t.shape, np.nan)
    for index in np.ndindex(t.shape):
        ti, ki, pi = t[index], k[index], price[index]
        if not np.isfinite([ti, ki, pi]).all() or ti <= 0 or ki <= 0:
            continue
        low = max(1-ki, 0)
        if pi < low-1e-12 or pi >= 1:
            continue
        if pi <= low+1e-14:
            result[index] = 0.
        elif pi <= bs_call(ti, ki, 10.):
            result[index] = brentq(lambda s: float(bs_call(ti, ki, s))-pi, 0., 10., xtol=1e-12)
    return result


def raw_svi(x, a, b, rho, m, sigma):
    """Raw SVI total variance at log-forward moneyness x."""
    z = np.asarray(x)-m
    return a+b*(rho*z+np.sqrt(z*z+sigma*sigma))


def _interp(t, times, values, short_scale=True, long_slope=0.):
    out = np.interp(t, times, values)
    if short_scale:
        out = np.where(t < times[0], values[0]*t/times[0], out)
    return np.where(t > times[-1], values[-1]+long_slope*(t-times[-1]), out)


def total_variance(model, t, k):
    """Broadcast surface total variance, including explicit extrapolation."""
    t, k = np.broadcast_arrays(np.asarray(t, float), np.asarray(k, float))
    if not np.isfinite(t).all() or not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError('finite times and positive finite strikes required')
    times = np.asarray(model['times'])
    p = np.asarray(model['parameters'])
    x = np.log(k)
    if model['kind'] == 'svi':
        slices = np.array([raw_svi(x, *row) for row in p])
        # Per-observation interpolation avoids confusing strike and time axes.
        flat = slices.reshape(len(times), -1)
        w = np.array([_interp(t.flat[j], times, flat[:, j]) for j in range(t.size)]).reshape(t.shape)
    else:
        if model['kind'] == 'ssvi':
            rho, eta = model['rho'], model['eta']
            theta = _interp(t, times, p, long_slope=p[-1]/times[-1])
            psi = eta*np.sqrt(np.maximum(theta, 0)/(1+np.maximum(theta, 0)))
        else:
            theta = _interp(t, times, p[:, 0], long_slope=p[-1, 0]/times[-1])
            psi = _interp(t, times, p[:, 2])
            product = _interp(t, times, p[:, 1]*p[:, 2])
            rho = np.divide(product, psi, out=np.zeros_like(psi), where=psi > 0)
        w = .5*(theta+rho*psi*x+np.sqrt((psi*x+theta*rho)**2+theta**2*(1-rho*rho)))
    return np.where(t <= 0, 0., w)


def price_surface(model, t, k):
    return _call_w(k, total_variance(model, t, k))


def _constraints(model):
    p = np.asarray(model['parameters'])
    if model['kind'] == 'svi':
        a, b, rho, m, s = p.T
        margins = [a+b*s*np.sqrt(1-rho*rho), 2-b*(1+np.abs(rho))]
        x = np.linspace(-1., 1., 61)
        w = np.array([raw_svi(x, *row) for row in p])
        z = x[None, :]-m[:, None]
        root = np.sqrt(z*z+s[:, None]**2)
        wp = b[:, None]*(rho[:, None]+z/root)
        wpp = b[:, None]*s[:, None]**2/root**3
        safe = np.maximum(w, 1e-12)
        g = (1-x*wp/(2*safe))**2-wp**2/4*(1/safe+.25)+wpp/2
        margins += [g.ravel(), np.diff(w, axis=0).ravel()]
    elif model['kind'] == 'ssvi':
        rho, eta = model['rho'], model['eta']
        margins = [np.diff(p), np.array([4-eta*eta*(1+abs(rho)), 4-eta*(1+abs(rho))])]
    else:
        theta, rho, psi = p.T
        margins = [np.diff(theta), np.diff(psi), 4-psi*(1+abs(rho)),
                   4*theta-psi*psi*(1+abs(rho)), np.diff(psi)-abs(np.diff(rho*psi))]
    return np.concatenate([np.asarray(v).ravel() for v in margins])


def _grid_check(model, observed_k):
    times = np.asarray(model['times'])
    tg = np.unique(np.r_[0., times, np.linspace(times[0]/10, times[-1]*1.25, 81)])
    kg = np.linspace(min(.5, float(np.min(observed_k))*.9), max(1.5, float(np.max(observed_k))*1.1), 241)
    c = price_surface(model, tg[:, None], kg[None, :])
    slopes = np.diff(c, axis=1)/np.diff(kg)
    checks = {'call_lower': float(np.min(c-np.maximum(1-kg, 0))),
              'call_upper': float(np.min(1-c)),
              'strike_monotonic': float(np.min(-slopes)),
              'strike_slope_lower': float(np.min(slopes+1)),
              'strike_convex': float(np.min(np.diff(slopes, axis=1))),
              'calendar': float(np.min(np.diff(c, axis=0)))}
    return {'valid': bool(np.isfinite(c).all() and min(checks.values()) >= -1e-7),
            'margins': checks, 'time_range': [float(tg[0]), float(tg[-1])],
            'strike_range': [float(kg[0]), float(kg[-1])],
            'shape': [len(tg), len(kg)], 'scope': 'finite_grid_only'}


def fit_surface(kind, ttm, k, prices, seed=20261112, starts=5):
    """Fit equal-weight normalized price MSE, five deterministic SLSQP starts.

    Rates and dividends must already be removed by the caller. All three models
    use the same quotes and objective; starts other than five are rejected.
    """
    tic = time.perf_counter()
    kind = kind.lower()
    if kind not in ('svi', 'ssvi', 'essvi') or starts != 5:
        raise ValueError('kind must be svi/ssvi/essvi and starts must be 5')
    t, k, prices = np.broadcast_arrays(np.asarray(ttm, float), np.asarray(k, float), np.asarray(prices, float))
    t, k, prices = t.ravel(), k.ravel(), prices.ravel()
    if not t.size or not np.isfinite([t, k, prices]).all() or np.any(t <= 0) or np.any(k <= 0):
        raise ValueError('nonempty finite data, positive times and strikes required')
    if np.any(prices < np.maximum(1-k, 0)-1e-12) or np.any(prices >= 1):
        raise ValueError('call prices outside normalized bounds')
    times = np.unique(t)
    n = len(times)
    iv = implied_vol(t, k, prices)
    useful = iv[np.isfinite(iv) & (iv > 0)]
    level = float(np.median(useful)) if useful.size else .2
    level = np.clip(level, .03, 2.)
    theta = np.clip(level**2*times, 1e-6, 4.)
    if kind == 'svi':
        initial = np.column_stack([theta, np.full(n, .001), np.zeros(n), np.zeros(n), np.full(n, .2)]).ravel()
        bounds = [(-2., 4.), (0., 2.), (-.98, .98), (-1., 1.), (.005, 2.)]*n
    elif kind == 'ssvi':
        initial = np.r_[theta, 0., .01]
        bounds = [(1e-8, 4.)]*n+[(-.98, .98), (0., 2.)]
    else:
        initial = np.column_stack([theta, np.zeros(n), .01*np.sqrt(theta)]).ravel()
        bounds = [(1e-8, 4.), (-.98, .98), (1e-8, 2.)]*n

    def unpack(z):
        model = {'kind': kind, 'times': times.tolist()}
        if kind == 'ssvi':
            model.update(parameters=z[:n].tolist(), rho=float(z[n]), eta=float(z[n+1]), gamma=.5)
        else:
            model['parameters'] = z.reshape(n, 5 if kind == 'svi' else 3).tolist()
        return model

    def objective(z):
        return float(np.mean((price_surface(unpack(z), t, k)-prices)**2))

    rng = np.random.default_rng(seed)
    candidates = []
    records = []
    for j in range(starts):
        z = initial.copy()
        if j:
            z *= rng.uniform(.75, 1.25, len(z))
            if kind == 'ssvi':
                z[n:] = [rng.uniform(-.7, .3), rng.uniform(.05, .8)]
            else:
                z.reshape(n, -1)[:, 2 if kind == 'svi' else 1] = rng.uniform(-.6, .2)
        z = np.clip(z, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        result = minimize(lambda v: 1e4*objective(v), z, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': lambda v: _constraints(unpack(v))},
                          options={'maxiter': 350, 'ftol': 1e-10})
        model = unpack(result.x)
        minimum = float(np.min(_constraints(model)))
        loss = objective(result.x)
        feasible = bool(minimum >= -1e-7 and np.isfinite(result.fun) and np.isfinite(loss))
        grid = _grid_check(model, k)
        qualified = bool(result.success and feasible and grid['valid'])
        records.append({'success': bool(result.success), 'feasible': bool(feasible),
                        'qualified': qualified, 'grid_valid': bool(grid['valid']),
                        'constraint_min': minimum, 'grid_margins': grid['margins'],
                        'objective': loss, 'message': str(result.message),
                        'optimizer_status': int(result.status), 'iterations': int(result.nit)})
        candidates.append((not qualified, loss, model, minimum, bool(result.success), grid, j))
    rejected, loss, model, minimum, success, grid, selected = min(candidates, key=lambda item: item[:2])
    model['diagnostics'] = {'valid': not rejected,
                            'optimizer_success': success, 'constraint_min': minimum,
                            'objective': loss, 'seconds': time.perf_counter()-tic,
                            'seed': int(seed), 'starts': starts, 'attempts': records, 'grid': grid,
                            'qualified_candidates': sum(row['qualified'] for row in records),
                            'selected_start': int(selected),
                            'selection_rule': 'minimum_price_MSE_among_optimizer_success_financial_and_grid_valid_candidates;minimum_failed_MSE_if_none',
                            'objective_definition': 'equal_weight_normalized_call_price_MSE',
                            'global_arbitrage_certified': False}
    return model


def summary(model):
    """30/365-year ATM IV and IV(1.05)-IV(.95), with validity flags."""
    t = 30/365
    iv = implied_vol(t, [1., 1.05, .95], price_surface(model, t, [1., 1.05, .95]))
    valid = model.get('diagnostics', {}).get('valid', False)
    atm_ok = bool(valid and np.isfinite(iv[0]) and iv[0] > 0)
    skew_ok = bool(valid and np.isfinite(iv[1:]).all() and np.all(iv[1:] > 0))
    return {'atm_iv_30d': float(iv[0]) if atm_ok else None,
            'skew_30d': float(iv[1]-iv[2]) if skew_ok else None,
            'atm_valid': atm_ok, 'skew_valid': skew_ok}
