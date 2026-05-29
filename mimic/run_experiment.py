# primary imports
import os, sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from pathlib import Path

# third party imports
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# local imports
from upocp import UPOCP
from pogo import POGO
from gcaci import GCACI
from utils import evaluate_method

# Load data
df = pd.read_csv('test_predictions_50.csv')

# drop irrelavant columns
col_sums = df.iloc[:, 3:].sum(numeric_only=True)
keep_cols = list(df.columns[:3]) + col_sums[col_sums >= 100].index.tolist()
df_filtered = df.loc[:, keep_cols]

# Extract conformity scores and group membership
y = df_filtered["true_los_icu"].values
y_hat = df_filtered["predicted_los_icu"].values
S = np.abs(y - y_hat)
C = df_filtered.iloc[:, 3:].values
num_groups = C.shape[1]
total_groups = num_groups + 1

# Path to save results
outdir = Path("results")
outdir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    alpha_grid = np.linspace(0.01, 0.25, 49)  # edit grid density as you like
    T = C.shape[0]

    for alpha in alpha_grid:
        print(f"Running for alpha={alpha:.3f}")

        # Instantiate methods
        methods = {
            "up-ocp": UPOCP(alpha=alpha),
            "pogo": POGO(num_groups=num_groups+1, alpha=alpha, binary_groups=True),
            "gcaci": GCACI(num_groups=num_groups+1, alpha=alpha, lr=1)
        }

        for i in range(T):
            S_t = S[i]
            c_t = np.concatenate([[1], C[i]])

            # up-ocp
            methods["up-ocp"].update(S_t)

            # pogo
            methods["pogo"].update(S_t, c_t)

            # gcaci
            methods["gcaci"].update(S_t, c_t)
        
        group_hist = methods['pogo'].group_history
        methods_payload = {name: evaluate_method(m, group_hist, alpha) for name, m in methods.items()}
        results = {0: {'methods': methods_payload}}

        joblib.dump(results, outdir / f"alpha={alpha:g}.joblib", compress=3)

        
