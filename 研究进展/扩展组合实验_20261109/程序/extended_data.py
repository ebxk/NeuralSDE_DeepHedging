"""Read frozen inputs and construct causal path tensors for the expanded study."""
from pathlib import Path

import numpy as np
import torch

from contract_ledger import EVENTS, ledger
from hedge_experiment import features, read_paths
from train import make_real
from extended_curve import bs_delta, reconstructed_delta


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return dict(file)


def historical(common):
    days = common["return_dates"]
    ret = common["returns"]
    factors = np.full((len(days), 2), np.nan)
    ix = np.searchsorted(days, common["real_dates"])
    assert np.array_equal(days[ix], common["real_dates"])
    factors[ix] = common["real_xi"]
    rv = np.full(len(days), np.nan)
    rv[19:] = np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(ret, 20)**2, axis=-1))
    tape = np.column_stack([20*ret, np.diff(factors, axis=0, prepend=np.nan),
                            4*np.diff(rv, prepend=np.nan)]).astype("f")
    return days, tape


def aligned_history(index, tape):
    grid = index[..., None] + np.arange(-9, 1)
    out = tape[grid.clip(0)]
    return np.where((grid >= 0)[..., None], out, np.nan)


def batch(data, rows, history=False):
    result = {key: value[rows] for key, value in data.items() if key != "tape"}
    if history:
        windows = data["tape"][rows].unfold(1, 10, 1).transpose(-1, -2)
        good = windows.isfinite()
        result["history"] = torch.where(good, windows, 0)
        result["history_valid"] = good
    return result


class Inputs:
    def __init__(self, episode_root, directory, rep, fold):
        self.episode_file = episode_root/"episode_days.csv"
        self.directory = directory
        self.rep = load_npz(rep)
        self.common = load_npz(directory/"common.npz")
        self.folds = fold
        self.paths = dict(zip(fold, read_paths(self.episode_file, fold)))
        self.calendar, self.tape = historical(self.common)
        self.real = {}
        self._sim = {}

    def simulation(self, market, need_delta=False):
        name = "single" if market == "single" else "pool"
        if name not in self._sim:
            d = load_npz(self.directory/f"{name}.npz")
            s, xi = d["spot"], d["xi"]
            t, warm = self.common["time"], self.common["warm"]
            strike = self.common["strike"].astype("f")
            tau = t[:, -1:] - t
            returns = np.diff(np.log(s), axis=1)
            rv = np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(
                np.concatenate([warm, returns], axis=1), 20, axis=1)**2, axis=-1))
            active = np.diff(t, axis=1) > 0
            f = features(s, self.common["strike"][:, None], tau, rv,
                         np.concatenate([xi, d["quality"]], axis=-1))[:, :-1].copy()
            origins = np.random.default_rng(104).integers(len(self.paths[self.folds[0]]), size=len(s))
            start_dates = np.array([self.paths[self.folds[0]][i][0]["date"] for i in origins])
            ix = np.searchsorted(self.calendar, start_dates)
            assert np.array_equal(self.calendar[ix], start_dates)
            # The unchanged simulator samples episode groups in integer ID order.
            assert np.allclose(s[:, 0], [float(self.paths[self.folds[0]][i][0]["spot"]) for i in origins])
            assert np.allclose(strike, [float(self.paths[self.folds[0]][i][0]["strike"]) for i in origins])
            warm_history = aligned_history(ix, self.tape)
            future = np.concatenate([20*returns[..., None], np.diff(xi, axis=1),
                                     4*np.diff(rv, axis=1)[..., None]], axis=2)
            future = np.where(active[..., None], future, np.nan)
            tape = np.concatenate([warm_history, future[:, :-1]], axis=1).astype("f")
            self._sim[name] = (dict(f=torch.from_numpy(f), spot=torch.from_numpy(s),
                unit=torch.ones_like(torch.from_numpy(s)), active=torch.from_numpy(active),
                premium=torch.from_numpy(self.common["premium"].astype("f")),
                strike_end=torch.from_numpy(strike), tape=torch.from_numpy(tape)),
                d["member"], xi, tau)
        data, ids, xi, tau = self._sim[name]
        if need_delta and "delta" not in data:
            data["delta"], data["delta_valid"] = reconstructed_delta(
                data["spot"].numpy()[:, :-1], self.common["strike"][:, None],
                tau[:, :-1], xi[:, :-1], self.rep, self.common["scale"])
        return data, ids

    def observed(self, year, need_delta=False, teacher=False):
        if teacher and year != self.folds[0]:
            raise ValueError("Teacher labels can only use the training fold")
        paths = self.paths[year]
        if year not in self.real:
            f, _ = make_real(paths, self.common, self.episode_file)
            n, h, _ = f.shape
            shape = (n, h+1)
            s, u, k, quote = (np.empty(shape, dtype="d") for _ in range(4))
            tau = np.zeros(shape)
            active = np.zeros((n, h), dtype=bool)
            tape = np.full((n, h+9, 4), np.nan, dtype="f")
            event = {key: np.zeros(shape, dtype="d" if key == "record_amount" else "int64")
                     for key in ("record_amount", "record_event", "ex_event", "pay_event")}
            for i, path in enumerate(paths):
                size = len(path)
                days = np.array([row["date"] for row in path])
                ix = np.searchsorted(self.calendar, days)
                assert np.array_equal(self.calendar[ix], days)
                for arr, key in [(s, "spot"), (u, "unit"), (k, "strike"), (quote, "option_close")]:
                    arr[i, :size] = [float(row[key]) for row in path]
                    arr[i, size:] = arr[i, size-1]
                tau[i, :size] = (days[-1].astype("datetime64[D]")-days.astype("datetime64[D]")).astype(float)/365
                active[i, :size-1] = True
                tape[i, :10] = aligned_history(ix[0], self.tape)
                tape[i, 10:size+8] = self.tape[ix[1:-1]]
                for j, (ex, record, pay, amount, _) in enumerate(EVENTS, 1):
                    for name, date in (("record_event", record), ("ex_event", ex), ("pay_event", pay)):
                        event[name][i, :size][days == date] = j
                    event["record_amount"][i, :size][days == record] = float(amount)
            data = dict(f=f, spot=torch.from_numpy(s), unit=torch.from_numpy(u),
                premium=torch.from_numpy(quote[:, 0].copy()), strike_end=torch.from_numpy(k[:, -1].copy()),
                active=torch.from_numpy(active), tape=torch.from_numpy(tape),
                **{key: torch.from_numpy(val) for key, val in event.items()})
            self.real[year] = data, k, tau, quote
        data, k, tau, quote = self.real[year]
        if need_delta and "delta" not in data:
            data["delta"], data["delta_valid"] = reconstructed_delta(
                data["spot"].numpy()[:, :-1], k[:, :-1], tau[:, :-1],
                data["f"].numpy()[:, :, 3:5], self.rep, self.common["scale"])
        if teacher and "teacher" not in data:
            data["teacher"], data["teacher_valid"] = bs_delta(
                data["spot"][:, :-1], k[:, :-1], quote[:, :-1], tau[:, :-1])
            data["teacher_valid"] &= data["active"]
        return paths, data


def books(paths, ratios, gate, cost):
    result = []
    for i, path in enumerate(paths):
        entries = ledger(path, map(float, ratios[i, :len(path)-1]), float(path[0]["option_close"]), cost/10000)
        for j, row in enumerate(entries):
            row["gate_weight"] = float(gate[i, j]) if gate is not None and j < len(path)-1 else ""
        result.append(entries)
    return result

