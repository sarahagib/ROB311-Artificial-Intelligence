"""
    Problem 3 Template file
"""
import random
import math

import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for a problem setup given by the "RRT_dubins_problem" class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file "rrt_planning.py". Your implementation
   can be tested by running "RRT_dubins_problem.py" (see the "main()" function).
2. Read all class and function documentation in "RRT_dubins_problem.py" carefully.
   There are plenty of helper functions in the class that you should use.
3. Your solution must meet all the conditions specificed below.
4. Below are some DOs and DONTs for this problem.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random points
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out related issues and will be generously set.
2. The planning function must return a list of nodes that represent a collision free path
   from the start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must be a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation of the node to understand the terminology)
3. The returned path should be a valid list of nodes with a Dubins-style path connecting the nodes. 
   i.e. the list should have the start node at index 0 and goal node at index -1. 
   For all other indices i in the list, the parent node for node i should be at index i-1,  
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   "RRT_dubins_problem.map_area"

DO(s) and DONT(s)
-------------------
1. DO rename the file to rrt_planning.py for submission.
2. Do NOT change change the "planning" function signature.
3. Do NOT import anything other than what is already imported in this file.
4. We encourage you to write helper functions in this file in order to reduce code repetition
   but these functions can only be used inside the "planning" function.
   (since only the planning function will be imported)
"""

def planning(rrt_dubins, display_map=False):
    """
        Execute RRT planning using dubins-style paths. Make sure to populate the node_lis

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """
    # Fix Randon Number Generator seed
    random.seed()

    # LOOP for max iterations
    i = 0
    yaw_lim = [np.deg2rad(-180), np.deg2rad(+180)]
    new_node = None
    
    while i < rrt_dubins.max_iter:
        i += 1

        # Generate a random vehicle state (x, y, yaw)
        
        #purely random search did not yield result before max iteration was met. 
        #added condition below 
         
        if i % 25 == 0 : 
            x = rrt_dubins.goal.x
            y = rrt_dubins.goal.y 
            yaw = rrt_dubins.goal.yaw
        else:
            x = random.uniform(rrt_dubins.x_lim[0]+0.001, rrt_dubins.x_lim[1]-0.001)
            y = random.uniform(rrt_dubins.y_lim[0]+0.001, rrt_dubins.y_lim[1]-0.001)
            yaw = random.uniform(yaw_lim[0], yaw_lim[1]) 
        
        vehicle_state = rrt_dubins.Node(x, y, yaw)
        
        # Find an existing node nearest to the random vehicle state
        nearest_node = None 
        nearest_dist = np.inf
        
        for node in rrt_dubins.node_list:
            curr_dist = rrt_dubins.calc_new_cost(node, vehicle_state) - node.cost
            
            if curr_dist < nearest_dist:
                nearest_node = node
                nearest_dist = curr_dist 
        
        new_node = rrt_dubins.propogate(nearest_node, vehicle_state) 
        
        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        if rrt_dubins.x_lim[0] <= new_node.x <= rrt_dubins.x_lim[1] and rrt_dubins.y_lim[0] <= new_node.y <= rrt_dubins.y_lim[1] :
            if rrt_dubins.check_collision(new_node):
                #print("got here")
                rrt_dubins.node_list.append(new_node) # Storing all valid nodes
            #else:
                #print("collision occured")
        else: 
            continue 

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map == True:
            rrt_dubins.draw_graph()

        # Check if new_node is close to goal
        if new_node.is_state_identical(rrt_dubins.goal) == True:
            print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
            #found_node = new_node
            break

    if i == rrt_dubins.max_iter:
        print('reached max iterations')
        return [] 

    # Return path, which is a list of nodes leading to the goal
    path = []
    curr = new_node 
    while curr is not None: 
        path.append(curr)
        curr = curr.parent
    
    path.reverse()
    return path

