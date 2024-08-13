import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    #SET UP:
    #get initial and goal state, define initial node
    initState = problem.init_state
    goalState = problem.goal_states[0]
    initNode = Node(None, initState, None, 0)

    #check if the initial state is the goal state, if yes return it immediately
    if initState == goalState:
        return initState, num_nodes_expanded, max_frontier_size

    #define set of explored states
    exploredStates = set()

    #define frontier as a priority queue so that we can prioritize by cost.
    frontier = queue.PriorityQueue()

    #CHECK FIRST NODE:
    frontier.put((0, initNode))
    num_nodes_expanded += 1

    while frontier.empty() != True:

        max_frontier_size = max(max_frontier_size, frontier.qsize())

        #unpack highest priority element into cost and node components
        currCost, currNode = frontier.get()

        #if current state is the goal state, we want to trace the path and return
        if (currNode.state == goalState):
            path = problem.trace_path(currNode, initState)
            num_nodes_expanded += len(exploredStates)
            return path, num_nodes_expanded, max_frontier_size


        else:
            #otherwise, get a list of possible actions from our position in grid.
            currActions = problem.get_actions(currNode.state)

            for a in currActions:
                #for each action, get the child node and check if child node was explored.
                childNode = problem.get_child_node(currNode, a)

                if childNode.state not in exploredStates:
                    #if not explored, calculated cost and add to priority queue
                    cost = problem.heuristic(childNode.state)+currNode.path_cost
                    frontier.put((cost, childNode))
                    exploredStates.add(childNode.state)

    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.35
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 10
    N = 10
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))

    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS