# primary imports
import os, sys
import argparse
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# third-party imports
import joblib
import numpy as np
import pandas as pd

# local imports
from upocp import UPOCP
from pogo import POGO
from gcaci import GCACI
from utils import evaluate_method


def save_alpha_joblib(alpha: float, result: dict, stock: str):
    outdir = Path("results") / stock   # <-- results/stock_name/
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, outdir / f"alpha={alpha:g}.joblib", compress=3)


def load_S_C(stock, folder="data/processed_data"):
    df = pd.read_csv(os.path.join(folder, f"{stock}_forecast.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    monday_idx = df.index[df["date"].dt.weekday == 0]
    if len(monday_idx) == 0:
        raise ValueError("No Monday found in the dataset to anchor td_idx.")
    anchor_pos = monday_idx[0]

    df["td_idx"] = (df.index - anchor_pos) + 1
    df = df[df["td_idx"] >= 1].copy()
    df.insert(1, "td_idx", df.pop("td_idx"))

    for m in range(2, 21):
        df[f"div_by_{m}"] = (df["td_idx"] % m == 0).astype(float)

    col = "conformity_score"
    cols = df.columns.tolist()
    cols.insert(3, cols.pop(cols.index(col)))
    df = df[cols]

    S = df["conformity_score"].to_numpy()
    C = df.iloc[:, 5:].to_numpy()
    return S, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run methods for a given stock.")
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="Stock symbol, e.g. AAPL, MSFT, TSLA"
    )
    args = parser.parse_args()

    stock = args.stock
    alpha_grid = np.linspace(0.01, 0.25, 49)

    outdir = Path("results", stock)
    outdir.mkdir(parents=True, exist_ok=True)

    S, C = load_S_C(stock=stock)
    T = C.shape[0]
    num_groups = C.shape[1]
    total_groups = num_groups + 1

    for alpha in alpha_grid:
        print(f"Running {stock} alpha={alpha:.3f}")

        # Instantiate methods
        methods = {
            "up-ocp": UPOCP(alpha=alpha),
            "pogo": POGO(num_groups=num_groups + 1, alpha=alpha, binary_groups=True),
            "gcaci": GCACI(num_groups=num_groups + 1, alpha=alpha, lr=1)
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

            group_hist = methods["pogo"].group_history

        methods_payload = {name: evaluate_method(m, group_hist, alpha) for name, m in methods.items()}
        results = {0: {'methods': methods_payload}}

        joblib.dump(results, outdir / f"alpha={alpha:g}.joblib", compress=3)