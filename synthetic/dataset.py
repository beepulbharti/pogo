# primary imports
import math
from dataclasses import dataclass
import numpy as np

@dataclass
class BrownianConfig:
    a: float
    b: float
    T: int
    num_groups: int
    bad_frac: float
    seed: int = 0

    # membership probabilities
    p_good: float = 0.25
    p_bad: float = 0.05

    # GOOD groups: iid mean-0 noise (gated by membership)
    good_sigma: float = 0.01

    # BAD groups: AR(1) / random-walk special case
    bad_lambda: float = 0.0     # 0 => random walk, >0 => mean reverting AR(1)
    bad_mean: float = 0.0
    bad_mu: float = 0.0
    bad_sigma: float = 0.0
    bad_bias_init: float = 0.0

    # Optional: hard bound S to [0,1] (otherwise clip below at 0)
    constrain: bool = True

    # Changepoint
    use_changepoint: bool = False
    changepoint_t: int = 0
    changepoint_amp: float = 0.0
    cp_bad_idx: int = 0  # index among BAD groups: 0,...,B-1

    # # Quadratic trend for a single bad group (gated by membership)
    use_quad_trend: bool = False
    quad_mult_delta: float = 0.5
    quad_bad_idx: int = 0   # index among BAD groups: 0,...,B-1 (default 0 like cp_bad_idx)
    quad_C: float = 0.0     # adds quad_C * (t/T)^2 when that bad group is active

    soft_membership: bool = False
    kappa: float = 0.1

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.num_bad = int(math.ceil(self.num_groups * self.bad_frac))
        self.num_good = self.num_groups - self.num_bad
        if self.bad_lambda < 0:
            raise ValueError("bad_lambda must be >= 0 (0 => RW, >0 => mean-reverting).")

    def generate(self):
        rng = self.rng
        T, k = self.T, self.num_groups
        G, B = self.num_good, self.num_bad

        # memberships C (T x k)
        C = np.empty((T, k), dtype=np.float32)

        if self.soft_membership:
            if G > 0:
                a, b = self.kappa * self.p_good, self.kappa * (1.0 - self.p_good)
                C[:, :G] = rng.beta(a, b, size=(T, G)).astype(np.float32)
            if B > 0:          # assumed in (0,1)
                a, b = self.kappa * self.p_bad, self.kappa * (1.0 - self.p_bad)
                C[:, G:] = rng.beta(a, b, size=(T, B)).astype(np.float32)
        else:
            if G > 0:
                C[:, :G] = (rng.random((T, G)) < self.p_good).astype(np.int8)
            if B > 0:
                C[:, G:] = (rng.random((T, B)) < self.p_bad).astype(np.int8)

        # baseline
        base = rng.beta(self.a, self.b, size=T)
        self.base = base

        # good_stuff (iid, mean 0), gated by memberships
        if G > 0 and self.good_sigma > 0:
            eps_good = self.good_sigma * rng.uniform(-1, 1, size=(T, k))
            good_stuff = (eps_good * C).sum(axis=1)
        else:
            good_stuff = np.zeros(T, dtype=float)

        # bad_stuff: AR(1) per bad group, gated by bad memberships
        if B > 0:
            lam = self.bad_lambda

            b_bad = np.empty((T, B), dtype=float)
            b_bad[0] = self.bad_bias_init

            for t in range(1, T):
                b_bad[t] = (
                    (1 - lam) * b_bad[t - 1]
                    + lam * self.bad_mean
                    + self.bad_mu
                    + self.bad_sigma * rng.uniform(-1, 1, size=B)
                )

            bad_active = C[:, G:]  # (T,B)
            bad_stuff = (b_bad * bad_active).sum(axis=1)

            # Changepoint shift for one bad group (gated by membership)
            if self.use_changepoint and self.changepoint_amp != 0.0:
                if not (0 <= self.cp_bad_idx < B):
                    raise ValueError(
                        f"cp_bad_idx must be in [0, {B-1}] but got {self.cp_bad_idx}."
                    )
                z = (np.arange(T) >= self.changepoint_t).astype(float)
                bad_stuff = bad_stuff + self.changepoint_amp * z * bad_active[:, self.cp_bad_idx]

        # Quadratic trend for one bad group with multiplicative noise: quad * (1 + delta * U_t),
        # where U_t ~ Uniform(-1, 1) so the multiplier has mean 1 (unbiased).
        if self.use_quad_trend and self.quad_C != 0.0:
            if not (0 <= self.quad_bad_idx < B):
                raise ValueError(f"quad_bad_idx must be in [0, {B-1}] but got {self.quad_bad_idx}.")

            t = np.arange(T, dtype=float)          # 0,...,T-1
            quad = self.quad_C * (t / T) ** 2      # deterministic quadratic trend

            delta = float(self.quad_mult_delta)
            if delta < 0 or delta > 1:
                raise ValueError("quad_mult_delta must be in [0, 1].")

            U = rng.uniform(-1.0, 1.0, size=T)     # mean 0
            quad_noisy = quad * (1.0 + delta * U)  # mean multiplier = 1

            bad_stuff = bad_stuff + quad_noisy * bad_active[:, self.quad_bad_idx]


        S = base + good_stuff + bad_stuff

        if self.constrain:
            S = np.clip(S, 0.0, 1.0)
        else:
            S = np.clip(S, 0.0, None)

        return S, C