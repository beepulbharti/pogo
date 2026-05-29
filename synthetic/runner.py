# primary imports
import os, sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# third-party imports
import numpy as np

# local imports
from upocp import UPOCP
from pogo import POGO
from gcaci import GCACI
from utils import evaluate_method

def run_one_simulation(cfg, *, alpha: float):
    T = cfg.T
    
    # Generate data
    S, C = cfg.generate()

    # Instantiate methods
    num_groups = cfg.num_groups
    methods = {
        "up-ocp": UPOCP(alpha=alpha),
        "pogo": POGO(num_groups=num_groups+1, alpha=alpha, binary_groups=True),
    }

    # GCACI family kept separate
    gc_aci = {f"gcaci_lr={lr}": GCACI(num_groups=num_groups+1, alpha=alpha, lr=lr)
            for lr in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]}

    # Run experiment
    C = np.column_stack([np.ones(C.shape[0]), C])
    for i in range(T):

        # S_t and c_t
        S_t = S[i]
        c_t = C[i]

        # marginal
        methods["up-ocp"].update(S_t)

        # group method
        methods["pogo"].update(S_t, c_t)

        # GCACI variants
        for m in gc_aci.values():
            m.update(S_t, c_t)

    group_hist = methods["pogo"].group_history 

    method_payload = {
        "up-ocp": evaluate_method(methods["up-ocp"], group_hist, alpha),
        "pogo": evaluate_method(methods["pogo"], group_hist, alpha),
        **{name: evaluate_method(m, group_hist, alpha) for name, m in gc_aci.items()},
    }

    return {
        "methods": method_payload,
    }