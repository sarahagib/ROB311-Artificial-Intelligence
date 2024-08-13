# part1_1.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 4

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleanign robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code
    
    """
    Personal Notes: 
        3 dimensional array breakdown : 
            each 2d element is the state we are moving from 
            each row in the state we are moving too
            each column (0, 1) represents and action (left, right) 
        while there are 2 possible moves, there are 3 possible results:
            going in intended direction (0.8)
            going in opposite direction (0.05)
            staying in place (0.15) 
        if we are in a terminal state to begin (0 or 5), we cannot move in 
        any direction regardless of action taken. so probability is 0. 
        
        wrote out this matrix on paper, found the common pattern. 
        [0.8, 0.05] [.15, .15] [0.05, 0.8]
    """
    #skip from state 0 and from state 5
    for i in range(1, 5):
        # moving to state on the left given action is [left, right]
        P[i][i-1] = [0.8, 0.05] 
        
        #staying in the same place regardless of action
        P[i][i] = [0.15, 0.15]
        
        #moving to state on the right given action is [left, right]
        P[i][i+1] = [0.05, 0.8] 
        
    #print(P)
    ## END: Student code
    return P