from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem


def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

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

    #SET UP:
        #basically, we are creating two BFS, one that start at the init state and another at the goal state.
        #need to set up twice the number of variables
    initState = problem.init_state
    goalState = problem.goal_states[0]

    initNode = Node(None, initState, None, 0)
    goalNode = Node(None, goalState, None, 0)

    initFrontier = deque()
    goalFrontier = deque()

    initExploredStates = {} #child:parent dictionaries
    goalExploredStates = {}

    pathFromInit = []
    pathToGoal = []

    initSet = set() #will be used to determine if an intersection between both paths has occured.
    goalSet = set()


    foundNode = False #set when intersecting node was found, and will stop the search
    foundPath = False #set when each path is found.

    #CHECK FIRST NODE OF EACH PATH:
    initFrontier.append(initNode)
    initExploredStates[initState] = None
    initSet.add(initState)

    goalFrontier.append(goalNode)
    #goalExploredStates[goalState] = None
    goalSet.add(goalState)

    num_nodes_expanded += 2

    """
    Design of bidirectional BFS :

    loop runs while both frontier > 0 and path not found in both directions :

        define temp queue to hold all neighbours (child) nodes that will be explored

        BFS loop for inital side - while the frontier not empty and have not found intersecting node:
            pop a neighbour
            add to setInit
            get action list of node.

            loop through action list :
                get child associated to each action.
                check if child is in goalSet, then found intersection
                or if child in goalFrontier, then node will be visited on goal side, and will become intersection.
                    -> break and run goal side to get to child.

                if not, check if child in initSet -> ie node was explored
                or if it's in temp queue that holds neighbours -> will eventually be explored
                    -> just continue, will get to node eventually

                if not in either
                    ->append child to temp queue and add to initSet and initExploredStates

            when done looping through action list
                -> load temp queue holding neighbours to initFrontier (official queue)
                -> update nr of expanded nodes

        BFS loop for goal side
            esentially repeat the process from the goal side
    """

    while (len(initFrontier)>0 and len(goalFrontier)>0 and not foundPath):

        max_frontier_size = max(max_frontier_size, len(initFrontier), len(goalFrontier))

        #BFS that starts at initial state
        tempQ = deque()
        while (len(initFrontier)>0 and foundNode!=True):
            currNode = initFrontier.popleft()
            initSet.add(currNode.state)
            currActions = problem.get_actions(currNode.state)

            a = 0
            while a <= (len(currActions) - 1):
                #regular BFS implementation
                childNode = problem.get_child_node(currNode, currActions[a])
                a += 1

                if (childNode.state in goalSet) or (childNode in goalFrontier):
                    #intersection was found or node is in goal-BFS queue and will become intersection.
                    intersectionState = childNode.state
                    initExploredStates[childNode.state] = currNode.state
                    initSet.add(intersectionState)

                    foundNode = True
                    foundPath = True
                    break
                else:
                    if (childNode.state in initSet) or (childNode in tempQ):
                        #node has been or will be explored
                        continue
                    else:
                        tempQ.append(childNode)
                        initSet.add(childNode.state)
                        initExploredStates[childNode.state] = currNode.state
        #when done looping through all the child-neighbour nodes or when intersecting state found
        initFrontier = tempQ
        num_nodes_expanded += len(tempQ) #add the increasing nr of child nodes
        if foundPath == True:
            break

        #BFS that starts at goal state
        tempQ = deque()
        while (len(goalFrontier)>0 and foundNode!=True):
            currNode = goalFrontier.popleft()
            goalSet.add(currNode.state)
            currActions = problem.get_actions(currNode.state)

            a = 0
            while a <= (len(currActions)-1):
                #regular BFS implementation
                childNode = problem.get_child_node(currNode, currActions[a])
                a+=1

                if (childNode.state in initSet) or (childNode in initFrontier):
                    intersectionState = childNode.state
                    goalExploredStates[childNode.state] = currNode.state
                    goalSet.add(intersectionState)
                    foundNode = True
                    foundPath = True
                    break
                else:
                    if (childNode.state in goalSet) or (childNode in tempQ):
                        continue
                    else:
                        tempQ.append(childNode) #will be checked eventually
                        goalSet.add(childNode.state)
                        goalExploredStates[childNode.state] = currNode.state
        goalFrontier = tempQ
        num_nodes_expanded += len(tempQ)
        if foundPath == True:
            break

    #two half paths were found and intersection node identified.
    try:
        #path found from init to intersection
        pathFromInit.append(intersectionState)

        currState = initExploredStates[intersectionState]
        while (currState != initState):
            pathFromInit.append(currState)
            currState = initExploredStates[currState]

        pathFromInit.append(initState)
        pathFromInit.reverse() #reverse to get correct order

        #path found from intersection to goal
        currGoalState = goalExploredStates[intersectionState]
        while (currGoalState != goalState):
            pathToGoal.append(currGoalState)
            currGoalState = goalExploredStates[currGoalState] #gives next node explored on path to goal

        pathToGoal.append(goalState)

        #combine to get full path
        path = pathFromInit + pathToGoal
        return path, num_nodes_expanded, max_frontier_size

    except KeyError:
        #if error occurs, as in no path is found between initial state and goal state
        #set path to None/empty.
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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('C:\\Users\\SarahAgib\\Desktop\\UofT-Year3B\\ROB311 - Artifical Intelligence\\lab1\\rob311_winter_2023_project_01_handout\\stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!