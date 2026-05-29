import numpy as np
from pogo import UP

class UPOCP:
    def __init__(self, burn_in=500, alpha=0.05):
        self.alpha = alpha
        self.t = 0
        self.burn_in = burn_in
        self.wealth = 1
        
        self.miscover_count = 0
        self.w1, self.w2 = 1, 1

        # Track histories
        self.radius_history = []
        self.conformity_history = []
        self.cover_history = []
        self.wealth_history = []

        # Initialize portfolio strategy
        self.portfolio_strategy = UP(num_groups=1)
    
    def update(self, S_t):
        self.t += 1

        # Update the lambda via UP
        self.lambd = (self.miscover_count + 0.5)/self.t

        # Update beta
        self.beta = (self.lambd - self.alpha)/(self.alpha*(1-self.alpha))

        # Update the radius using tau_t = W_{t-1} * beta_t
        self.radius = self.wealth * self.beta

        # Update wealth by evaluating subgradient
        subgrad_t = (S_t <= self.radius)*1 - (1-self.alpha)
        self.w1, self.w2 = 1 - subgrad_t/self.alpha, 1 + subgrad_t/(1-self.alpha)
        self.wealth *= (1 - self.beta * subgrad_t)

        # Update miscover count
        self.miscover_count += (self.radius < S_t)*1

        # If time > burn in, then start saving results
        if self.t > self.burn_in:
            self.radius_history.append(np.maximum(self.radius, 0))
            self.conformity_history.append(S_t)
            self.cover_history.append(float(self.radius >= S_t))
            self.wealth_history.append(self.wealth)