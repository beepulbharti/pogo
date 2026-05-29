import numpy as np

class GCACI:
    def __init__(self, num_groups, burn_in=500, alpha=0.05, lr=1):
        self.num_groups = num_groups
        self.alpha = alpha
        self.lr = lr
        self.t = 0
        self.burn_in = burn_in

        # theta is the parameter vector for linear radius model (radius = theta @ c_t)
        self.theta = np.zeros(num_groups)

        # Track histories
        self.radius_history = []
        self.conformity_history = []
        self.cover_history = []
        self.group_history = []
    
    def update(self, S_t, c_t):
        self.t += 1

        # Update radius
        self.radius = self.theta @ c_t
        
        # Update radius
        if S_t > self.radius:
            self.theta = self.theta + self.lr * (1-self.alpha) * c_t
        else:
            self.theta = self.theta - self.lr * self.alpha * c_t

        if self.t > self.burn_in:
            self.radius_history.append(np.maximum(self.radius, 0))
            self.conformity_history.append(S_t)
            self.group_history.append(c_t)
            self.cover_history.append(float(self.radius >= S_t))