# primary imports
from pathlib import Path

# third-party imports
import joblib
import numpy as np
import pandas as pd

def group_coverage_rates(cover_history, group_histories, *, empty=np.nan):
    cover = np.asarray(cover_history, dtype=np.float64)      # (T,)
    groups = np.asarray(group_histories, dtype=np.float64)   # (T,K)

    if cover.ndim != 1 or groups.ndim != 2 or groups.shape[0] != cover.shape[0]:
        raise ValueError("cover_history must be (T,) and group_histories must be (T,K)")

    counts = groups.sum(axis=0)          # (K,)
    sums = cover @ groups                # (K,) == sum_t cover[t] * groups[t,k]

    return np.divide(
        sums, counts,
        out=np.full_like(counts, empty, dtype=np.float64),
        where=counts != 0
    )

def _longest_zero_run(x):
    x = np.asarray(x, dtype=np.int8)
    max_len = 0
    cur = 0
    for v in x:
        if v == 0:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return int(max_len)

def radius_stats(r):
    r = np.asarray(r, dtype=float)
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "median": float(np.median(r)),
        "q75": float(np.quantile(r, 0.75)),
        "q90": float(np.quantile(r, 0.90)),
        "q95": float(np.quantile(r, 0.95)),
    }

def evaluate_method(method, group_history, alpha=0.05, top_k=5, empty=np.nan):
    covered = np.asarray(method.cover_history, dtype=np.int8)      # (T,)
    tau = np.asarray(method.radius_history, dtype=float)           # (T,)
    C = np.asarray(group_history, dtype=np.float64)                # (T,K)

    if covered.ndim != 1 or tau.ndim != 1 or C.ndim != 2 or C.shape[0] != covered.shape[0] or C.shape[0] != tau.shape[0]:
        raise ValueError("cover_history and radius_history must be (T,) and group_history must be (T,K) with same T")

    target = 1.0 - alpha
    T, K = C.shape

    # 1) Overall marginal coverage
    overall_cov = float(covered.mean())

    # 2) 5 worst group coverages by absolute gap from target
    group_cov = group_coverage_rates(covered, C, empty=empty)   # (K,)
    gaps = np.abs(group_cov - target)

    valid = ~np.isnan(group_cov)
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        gaps_valid = gaps[valid_idx]

        k_eff = min(top_k, len(valid_idx))
        sel = np.argpartition(-gaps_valid, k_eff - 1)[:k_eff]
        worst_groups = valid_idx[sel]
        worst_groups = worst_groups[np.argsort(-gaps[worst_groups])]

        worst = [
            {
                "group": int(g),
                "coverage": float(group_cov[g]),
                "gap": float(gaps[g]),
                "count": float(C[:, g].sum()),
            }
            for g in worst_groups
        ]
    else:
        worst = []

    # 3) Longest error sequence (marginal)
    longest_marginal = _longest_zero_run(covered)

    # 4) Longest error sequence among groups (and which group)
    longest_group = 0
    longest_group_id = None
    for g in range(K):
        mask = C[:, g] != 0
        if not np.any(mask):
            continue
        lg = _longest_zero_run(covered[mask])
        if lg > longest_group:
            longest_group = lg
            longest_group_id = g

    # 5) Set size statistics (tau stats)
    tau_stat = radius_stats(tau)

    return {
        "overall_coverage": overall_cov,
        "worst_groups": worst,  # length <= top_k
        "longest_error_seq_marginal": int(longest_marginal),
        "longest_error_seq_group": int(longest_group),
        "longest_error_seq_group_id": None if longest_group_id is None else int(longest_group_id),
        "set_size_stats": tau_stat,
    }

def print_results(results, alpha=0.05, max_groups_to_print=10):
    target = 1 - alpha

    print("=" * 60)
    print("COVERAGE")
    print("=" * 60)
    print(f"Overall coverage:  {results['overall_coverage']:.4f}  (target: {target:.2f})")
    print()

    print("=" * 60)
    print("WORST GROUPS (by |coverage - target|)")
    print("=" * 60)
    worst = results.get("worst_groups", [])
    if not worst:
        print("No valid groups (all empty / NaN).")
    else:
        for i, w in enumerate(worst[:max_groups_to_print], 1):
            print(f"{i:2d}. group={w['group']:<4d} "
                  f"coverage={w['coverage']:.4f}  "
                  f"gap={w['gap']:.4f}  "
                  f"count={w['count']:.0f}")
        if len(worst) > max_groups_to_print:
            print(f"... ({len(worst) - max_groups_to_print} more groups)")
    print()

    print("=" * 60)
    print("LONGEST ERROR SEQUENCE")
    print("=" * 60)
    print(f"Longest consecutive miscoverage (marginal): {results['longest_error_seq_marginal']}")
    lg = results.get("longest_error_seq_group", 0)
    gid = results.get("longest_error_seq_group_id", None)
    if gid is None:
        print("Longest consecutive miscoverage (within any group): n/a (no non-empty groups)")
    else:
        print(f"Longest consecutive miscoverage (within a group): {lg}  (group: {gid})")
    print()

    print("=" * 60)
    print("SET SIZE (TAU) STATISTICS")
    print("=" * 60)
    ts = results["set_size_stats"]
    print(f"Tau: mean={ts['mean']:.4f}, median={ts['median']:.4f}, q75={ts['q75']:.4f}, q90={ts['q90']:.4f}, q95={ts['q90']:.4f}")


def create_df(dgp_name):
    num = int((0.25 - 0.01) / 0.005) + 1
    alphas = np.linspace(0.01, 0.25, num)
    rows = []
    RESULTS_DIR = Path("results")
    outdir = RESULTS_DIR / dgp_name

    for alpha in alphas:
        
        alpha_min = f"{alpha:.12g}"          # e.g. 0.1, 0.01, 0.25
        alpha_strip = f"{alpha:.3f}".rstrip("0").rstrip(".")  # e.g. 0.1, 0.01, 0.25

        candidates = [
            outdir / f"alpha={alpha_min}.joblib",
            outdir / f"alpha={alpha_strip}.joblib",
            outdir / f"alpha={alpha:.2f}.joblib",
            outdir / f"alpha={alpha:.3f}.joblib",
        ]

        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            continue

        runs_dict = joblib.load(p)

        methods = set()
        for run in runs_dict.values():
            methods |= set(run["methods"].keys())

        for method in sorted(methods):
            set_means = []
            worst_covs = []
            longest_group_errs = []

            for run in runs_dict.values():
                md = run["methods"].get(method)
                if md is None:
                    continue

                set_means.append(md["set_size_stats"]["mean"])

                wg = md.get("worst_groups", [])
                worst_covs.append(wg[0]["coverage"] if len(wg) > 0 else np.nan)

                longest_group_errs.append(md["longest_error_seq_group"])

            if len(set_means) == 0:
                continue

            set_means = np.asarray(set_means, dtype=float)
            worst_covs = np.asarray(worst_covs, dtype=float)
            longest_group_errs = np.asarray(longest_group_errs, dtype=float)

            n_set = np.isfinite(set_means).sum()
            n_cov = np.isfinite(worst_covs).sum()
            n_lge = np.isfinite(longest_group_errs).sum()

            rows.append({
                "dgp": dgp_name,
                "alpha": alpha,
                "method": method,

                "avg_set_size_mean": float(np.nanmean(set_means)),
                "avg_set_size_mean_sem": float(np.std(set_means[np.isfinite(set_means)], ddof=1) / np.sqrt(n_set)) if n_set > 1 else 0.0,

                "avg_worst_group_coverage": float(np.nanmean(worst_covs)),
                "avg_worst_group_coverage_sem": float(np.std(worst_covs[np.isfinite(worst_covs)], ddof=1) / np.sqrt(n_cov)) if n_cov > 1 else 0.0,

                "avg_longest_group_error_seq": float(np.nanmean(longest_group_errs)),
                "avg_longest_group_error_seq_sem": float(np.std(longest_group_errs[np.isfinite(longest_group_errs)], ddof=1) / np.sqrt(n_lge)) if n_lge > 1 else 0.0,

                "n_runs": int(n_set),
            })

    return pd.DataFrame(rows).sort_values(["method", "avg_set_size_mean"]).reset_index(drop=True)