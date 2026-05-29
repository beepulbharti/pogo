# primary imports
from pathlib import Path

# third-party imports
import joblib
from joblib import Parallel, delayed
import numpy as np

# local imports
from exp_registry import build_cfg
from runner import run_one_simulation


# ---- settings ----
num = int((0.25 - 0.01) / 0.005) + 1
DGP_NAMES = [
            'bounded_no_changepoint',
            'bounded_changepoint',
             'unbounded_slow',
             'unbounded_moderate',
             'unbounded_extreme'
             ] 
T         = 50000
ALPHAS    = np.linspace(0.01, 0.25, num)
N_RUNS    = 50
SEED0     = 0
N_JOBS    = -1
NUM_GROUPS = 100
# ------------------

def one_run(dgp_name: str, run_id: int, alpha: float, num_groups: int):
    seed = SEED0 + run_id
    cfg = build_cfg(dgp_name, seed=seed, T=T, num_groups=num_groups)
    return run_id, run_one_simulation(cfg, alpha=alpha)

def save_alpha_joblib(dgp_name: str, num_groups: int, alpha: float, runs_dict: dict):
    outdir = Path("results") / f"k={num_groups}" / dgp_name
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(runs_dict, outdir / f"alpha={alpha:g}.joblib", compress=3)

def main():
    for dgp_name in DGP_NAMES:
        num_groups = NUM_GROUPS

        for alpha in ALPHAS:  # sequential over alpha
            print(f"Running {dgp_name} for alpha={alpha:.3f}")

            items = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(one_run)(dgp_name, run_id, float(alpha), num_groups)
                for run_id in range(N_RUNS)
            )
            runs_dict = dict(items)

            save_alpha_joblib(dgp_name, num_groups, float(alpha), runs_dict)

if __name__ == "__main__":
    main()