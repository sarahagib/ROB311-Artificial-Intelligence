from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

###
def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []


    #my code

    #set up of know conditions : goal state, initial state and starting node.
    goalState = problem.goal_states[0]
    initState = problem.init_state
    startingNode = Node(None, initState, None, 0)

    #define dictionary of explored states which will contain child:parent pairs
    #define queue that will hold neighbours to be checked.
    exploredStates = {}
    frontierQ = deque()


    #explore first node (first search)
    frontierQ.append(startingNode)
    exploredStates[initState] = None
    num_nodes_expanded += 1


    #define breadth first search loop
    while (len(frontierQ) > 0):
        '''
        Implement BFS with a FIFO queue.
        pop current node off queue
        ->if it's the goal state, reverse the path and return.
        -> if not:
            -> check it's neighbours (i.e. actions)
                ->if action (child) is goal state, add to path final path, reverse the path and return
                ->if action (child) already in frontier queue, continue
                ->if not, add to frontierQ
            -> increase max_frontier_size
        '''

        max_frontier_size = max(len(frontierQ), max_frontier_size)

        currNode = frontierQ.popleft()
        num_nodes_expanded += 1

        #if statement enacts when we have found the goal state.
        if currNode.state == goalState :
            path = problem.trace_path(currNode, initState)
            return path, num_nodes_expanded, max_frontier_size

        #if current state is not goal state.
        #1. get all actions associate to current state
        currActions = problem.get_actions(currNode.state)

        #2. for each action, get child node
        for a in currActions:
            childNode = problem.get_child_node(currNode, a)

            #two cases:
                #childNode already in queue
                #childNode not in queue and
                    #->need to check if it's the goal state
                    #->if not need to add it to the queue to be checked.

            if childNode.state in exploredStates:
                continue #will be eventually explored.

            else:
                exploredStates[childNode.state] = currNode #add to explored.

                if childNode.state == goalState:
                    #essentially repeat the process in the first if statement.
                    path = problem.trace_path(childNode, initState)
                    return path, num_nodes_expanded, max_frontier_size
                else:
                    frontierQ.append(childNode)

    #null case. no path to be found.
    path = []
    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('C:\\Users\\SarahAgib\\Desktop\\UofT-Year3B\\ROB311 - Artifical Intelligence\\lab1\\rob311_winter_2023_project_01_handout\\stanford_large_network_facebook_combined.txt',dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
