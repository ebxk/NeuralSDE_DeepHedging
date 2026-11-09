"""Fixed model registry; one key per distinct training job."""
from dataclasses import asdict, dataclass
from itertools import product


@dataclass(frozen=True)
class Job:
    end: int
    rep: str
    r: int
    t: str
    w: bool
    h: bool
    cost: int
    lam: float
    seed: int
    market: str
    policy: str
    family: str

    @property
    def key(self):
        return (f"{self.end}_R{self.r}_{self.rep}_{self.t}_W{int(self.w)}H{int(self.h)}_"
                f"{self.cost}bp_L{self.lam:g}_S{self.seed}_{self.market}_{self.policy}")

    def meta(self):
        return asdict(self)


def registry(cfg):
    result = {}

    def add(family, end, rep, r, t, w, h, cost=5, market="single", policy="wide"):
        for lam, seed in product(cfg["lambdas"], cfg["seeds"]):
            j = Job(end, rep, r, t, w, h, cost, lam, seed, market, policy, family)
            result.setdefault(j.key, j)

    reps = ("C0", "C1", "C2")
    bits = (False, True)
    for _, _, end in cfg["folds"]:
        for rep, r, t, w, h in product(reps, (0, 1), ("off", "anneal"), bits, bits):
            add("main", end, rep, r, t, w, h)
        for rep, r in product(reps, (0, 1)):
            add("narrow", end, rep, r, "off", False, False, policy="narrow")

    for r, enabled in product((0, 1), bits):
        for t in ("warm", "constant", "real"):
            add("teacher", 2025, "C1", r, t, enabled, enabled)
        for policy in ("flat", "gru"):
            add("history", 2025, "C1", r, "anneal" if enabled else "off", True, True, policy=policy)
        for policy in ("delta_input", "delta_residual"):
            add("delta", 2025, "C1", r, "anneal" if enabled else "off", enabled, enabled, policy=policy)
    for r, market in product((0, 1), ("single", "equal", "robust")):
        for policy in ("base", "wide", "state_gate", "reliable_gate", "capacity"):
            add("bridge", 2025, "C1", r, "anneal" if r else "off", bool(r), bool(r),
                market=market, policy=policy)
    for rep, r, enabled, cost in product(reps, (0, 1), bits, (0, 10)):
        add("cost", 2025, rep, r, "anneal" if enabled else "off", enabled, enabled, cost=cost)
    assert len(result) == 1776
    return result

