import json
from pathlib import Path
import sys
import numpy as np

def main(source, upstream, output, config):
    cfg = json.loads(Path(config).read_text(encoding="utf8"))
    np.random.seed(cfg["factor_seed"])
    sys.path.insert(0, upstream)
    from marketmodel.factors import DecodeFactor
    from cvxopt import matrix, solvers

    data = np.load(source)
    A, b, calls = (data[k] for k in ("A", "b", "calls"))
    train = data["dates"] < str(cfg["split_ends"][0]+1)
    center = calls[train].mean(axis=0)
    residual = calls[train] - center
    pca, scores = DecodeFactor.decode_pca_factor(residual, 1)
    remainder = residual - scores @ pca
    rhs = (b - A @ center)[:, None] - (A @ pca.T) @ scores.T
    sa, _, weights = DecodeFactor.decode_stcarb_factor(remainder, 1, 4, A, rhs)
    basis = np.vstack([pca, sa])
    W, bounds = A @ basis.T, b - A @ center
    assert np.min(-bounds) >= -1e-7
    states = (calls - center) @ np.linalg.pinv(basis)
    solvers.options["show_progress"] = False
    for i, state in enumerate(states):
        if np.min(W @ state - bounds) < -1e-8:
            fit = solvers.qp(matrix(np.eye(2)), matrix(-state), matrix(-W), matrix(-bounds))
            if fit["status"] != "optimal":
                raise ValueError(f"Projection failed: {data['dates'][i]}")
            states[i] = np.asarray(fit["x"]).ravel()
    slack = np.min(states @ W.T - bounds)
    assert slack >= -1e-7
    print("Training dates:", sum(train), "minimum slack:", slack)
    np.savez_compressed(output, dates=data["dates"], spot=data["spot"], xi=states,
                        reconstruction_rmse=np.sqrt(np.mean((center+states@basis-calls)**2,axis=1)),
                        center=center, basis=basis, W=W, bounds=bounds, weights=weights)

if __name__ == "__main__":
    main(*sys.argv[1:])
