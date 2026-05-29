from dataclasses import replace
from dataset import BrownianConfig

BASE_CFG = BrownianConfig(
    a=1,
    b=20,
    T=50000,
    num_groups=25,
    bad_frac=0.05,
    seed=0,
)

DGP_OVERRIDES = {
    "bounded_no_changepoint": dict(
        constrain=True,
        bad_lambda=0.1,
        bad_mean=0.2,
        bad_mu=0.0,
        bad_sigma=1e-2,
        use_changepoint=False,
    ),
    "bounded_changepoint": dict(
        constrain=True,
        bad_lambda=0.1,
        bad_mean=0.2,
        bad_mu=0.0,
        bad_sigma=1e-2,
        use_changepoint=True,
        changepoint_amp=0.6,
    ),
    "unbounded_slow": dict(
        constrain=False,
        bad_lambda=0,
        bad_mean=0.2,
        bad_mu=1e-3,
        bad_sigma=1e-2,
        use_changepoint=False,
        use_quad_trend = True,
        quad_C = 5
    ),
    "unbounded_moderate": dict(
        constrain=False,
        bad_lambda=0,
        bad_mean=0.2,
        bad_mu=1e-3,
        bad_sigma=1e-2,
        use_changepoint=False,
        use_quad_trend = True,
        quad_C = 25
    ),
    "unbounded_extreme": dict(
        constrain=False,
        bad_lambda=0,
        bad_mean=0.2,
        bad_mu=1e-3,
        bad_sigma=1e-2,
        use_changepoint=False,
        use_quad_trend = True,
        quad_C = 100
    ),

}

def build_cfg(name: str, *, seed: int, T: int, num_groups: int) -> BrownianConfig:
    if name not in DGP_OVERRIDES:
        raise KeyError(f"Unknown DGP {name!r}. Options: {list(DGP_OVERRIDES)}")

    overrides = dict(DGP_OVERRIDES[name])

    if overrides.get("use_changepoint", False) and "changepoint_t" not in overrides:
        overrides["changepoint_t"] = T // 3

    if num_groups is not None:
        overrides["num_groups"] = num_groups

    return replace(BASE_CFG, T=T, seed=seed, **overrides)