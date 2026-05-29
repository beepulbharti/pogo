import numpy as np

class POGO:
    def __init__(self, num_groups, burn_in=500, alpha=0.05, binary_groups=False):
        self.num_groups = num_groups
        self.alpha = alpha
        self.t = 0
        self.burn_in = burn_in
        self.wealth = (1/num_groups) * np.ones(num_groups)

        self.miscover_counts = np.zeros(num_groups)
        self.group_counts = np.zeros(num_groups)
        self.binary_groups = binary_groups
        if binary_groups == False:
            self.w1, self.w2 = np.ones(num_groups), np.ones(num_groups)

        # Track histories
        self.radius_history = []
        self.conformity_history = []
        self.cover_history = []
        self.group_history = []
        self.wealth_history = []

        # Initialize portfolio strategy
        self.portfolio_strategy = UP(self.num_groups)

    def update(self, S_t, c_t):
        self.t += 1

        # Vectorized update of lambda with UP
        if self.binary_groups:
            self.lambd = (self.miscover_counts + 0.5)/(self.group_counts + 1)
        else:
            self.lambd = self.portfolio_strategy.next_weight(self.w1, self.w2)

        # Vectorized update of beta
        self.beta = (self.lambd - self.alpha)/(self.alpha*(1-self.alpha))

        # Vectorized update of theta in tau = theta @ c_t
        self.theta = self.wealth * self.beta

        # Update radius
        self.radius = self.theta @ c_t

        # Evaluate subgradient and update wealth
        Z_t = (S_t <= self.radius)*1 - (1-self.alpha)
        g_t = Z_t * c_t  # Subgradient of loss w.r.t. theta
        self.w1, self.w2 = 1 - g_t/self.alpha, 1 + g_t/(1-self.alpha)
        self.wealth *= (1 - self.beta * g_t)

        # Update miscover and group counts
        self.miscover_counts += (self.radius < S_t)*c_t
        self.group_counts += c_t

        if self.t > self.burn_in:
            self.radius_history.append(np.maximum(self.radius, 0))
            self.conformity_history.append(S_t)
            self.group_history.append(c_t)
            self.cover_history.append(float(self.radius >= S_t))
            self.wealth_history.append(self.wealth)

class UP:
    def __init__(self, num_groups, n_grid=20001, eps=1e-12, floor=1e-300, dtype=float):
        self.num_groups = int(num_groups)
        self.b = np.linspace(eps, 1.0 - eps, n_grid, dtype=dtype)
        self.floor = dtype(floor)

        p0 = 1.0 / (np.pi * np.sqrt(self.b * (1.0 - self.b)))
        p0 /= np.trapezoid(p0, self.b)

        self.p = np.tile(p0, (self.num_groups, 1)).astype(dtype, copy=False)

    def next_weight(self, w1, w2):
        """
        Update with realized gross returns (x1,x2) (shape (num_groups,))
        and return lambda for NEXT period (shape (num_groups,)).
        """
        w1 = np.asarray(w1, dtype=self.p.dtype).reshape(self.num_groups, 1)
        w2 = np.asarray(w2, dtype=self.p.dtype).reshape(self.num_groups, 1)

        r = self.b[None, :] * w1 + (1.0 - self.b[None, :]) * w2
        r = np.maximum(r, self.floor)

        self.p *= r
        self.p /= np.trapezoid(self.p, self.b, axis=1).reshape(self.num_groups, 1)

        return np.trapezoid(self.b[None, :] * self.p, self.b, axis=1)