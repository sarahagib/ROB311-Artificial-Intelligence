# part2.py: Project 4 Part 2 script
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
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states), 1))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    """
    Personal Notes: 
        -> heavily relied on textbook pseudocode 
        -> policy evaluation = givent a policy xi, calculate 
        Ui = Uxi, the utility of each state if xi were to be executed. 
        
        -> step 1 : evaluate policy. this process should update agent.utility 
        as the policy is evaluated. 
        
        -> step 2 : start the policy updating loop. 
            -> for each state s in S : 
                2.1: find the max value generated from each action in a over all 
                    possible next states 
                    max a of A(s) sum of s' P(s'|s, a)U[s'] 
                2.2: calculate the sum over all possible next states given the 
                    action decided in the policy
                    sum over s' P(s'| s, pi[s])U[s']
                2.3: if 2.1 > 2.2 replace the move in the policy with the better move.
        -> the algorithm terminates when the Utility function remains unchanged after 
        a policy improvement step occurs. 
        -> if unchanged = True, that means policy was changed. 
        -> if unchanged = False, that means policy was not changed, break out of loop. 
    """
    
    iter_count = 0 
    unchanged = True 
    #U = np.zeros([len(env.states), 1])
    
    while unchanged == True and iter_count <= max_iter: 
        iter_count += 1 
        unchanged = False 
        
        #step 1 : evaluate policy and update agent.utility
        U = np.copy(agent.utility) 
        for s in env.states:
            #for each state, identify action taken by policy 
            a = policy[s]
            policy_util_s = 0 
            for s_prime in env.states:
                #for each possible next state, calculate the utility give s, a
                policy_util_s += env.transition_model[s, s_prime, a] * U[s_prime]
            #update utility
            agent.utility[s] = env.rewards[s] + agent.gamma*policy_util_s
        
        #step 2
        for s in env.states: 
            #step 2.1 : code will be very similar to part 1.2 
            utility_by_action = [] 
            for a in env.actions:
                a_sum = 0 
                for s_prime in env.states: 
                    a_sum += env.transition_model[s, s_prime, a] * agent.utility[s_prime] 
                a_sum = float(a_sum) 
                utility_by_action.append(a_sum) 
            #step 2.2 
            utility_by_policyA = 0 
            policy_a = policy[s] 
            for s_prime in env.states:
                utility_by_policyA += env.transition_model[s, s_prime, policy_a] * agent.utility[s_prime]
            
            #step 2.3 
            utility_by_action_max = max(utility_by_action)
            if utility_by_action_max > utility_by_policyA : 
                policy[s] = np.argmax(utility_by_action)
                unchanged = True
    policy = policy.flatten()
    ## END: Student code

    return policy

