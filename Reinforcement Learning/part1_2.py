# part1_2.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 4

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states)
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    """
    Personal Notes: 
        -> heavily used pseudocode from textbook
    """
    u_prime = np.zeros([len(env.states), 1])
    threshold = eps * (1-agent.gamma)/agent.gamma 
    delta = np.inf 
    iter_count = 0 
    
    #repeat until delta < eps(1-gamma)/gamma 
    while delta>=threshold and iter_count <= max_iter:
        iter_count += 1 
        
        #u <- u'
        agent.utility[:] = u_prime   
        #delta <- 0 
        delta = 0 
        
        #for each state s in S do :
        for s in env.states:
            #create a list that will hold the summation over all possible states by action 
            utility_by_action = [] 
            
            #break down of max and summation part of update rule. 
            for a in env.actions:
                
                action_sum = 0
                for s_prime in env.states:
                    #action_sum += P(s'|s, a) * U[s']
                    action_sum += env.transition_model[s, s_prime, a] * agent.utility[s_prime] 
                    
                action_sum = float(action_sum)
                utility_by_action.append(action_sum)
                
            #U'[s] <- R[s] + gamma * max a in A(s) of utility by action
            u_prime[s] = env.rewards[s] + (agent.gamma * max(utility_by_action)) 
            
            if abs(u_prime[s] - agent.utility[s]) > delta: 
                delta = abs(u_prime[s] - agent.utility[s]) 
    
    #policy : choosing action that results in the greatest future reward (quality, utility). 
    for s in env.states:
        #for each state, get the utility value by action moving to each next state
        util_by_action = []
        for a in env.actions:
            a_sum = 0 
            for s_prime in env.states:
                a_sum += env.transition_model[s, s_prime, a] * agent.utility[s_prime] 
            a_sum = float(a_sum)
            util_by_action.append(a_sum)
        
        policy[s] = np.argmax(util_by_action)
  
    ## END Student code
    return policy