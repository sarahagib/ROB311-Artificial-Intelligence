# mdp_agent.py: Project 4
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 4

import numpy as np
from mdp_env import mdp_env

class mdp_agent:
    """
    Attributes -
        gamma:   (float) Discount factor, AIMA (4ed pg 649)
        utility: (np.array) The Utility matrix, AIMA (4ed pg 649)

    """
    def __init__(self, gamma: float):

        # Discount Factor
        assert 0.0 < gamma <= 1.0
        self.gamma = gamma

        # Utility Function
        self.utility = []
