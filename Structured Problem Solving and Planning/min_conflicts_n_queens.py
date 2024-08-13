import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000
    
    #start by mapping out the board, what places are occupied or conflicted. 
    
    #define a faster system, each index counts the number of conflicts in that row/diagonal. 
    #the same organization as the initialization is used but instead of dictionaries we will use lists for speed. 
    occupiedRows = [0]*N
    occupiedRightDiag = [0]*(N+N-1)
    occupiedLeftDiag = [0]*(N+N-1)
    
    for column in range(N):
        row = solution[column]    
        
        occupiedRows[row] += 1
        occupiedRightDiag[N-1-column-row] += 1
        occupiedLeftDiag[column-row] += 1
    
    
    for idx in range(max_steps):
        ## YOUR CODE HERE
        
        if idx == 999 :
            return [], -1
        
        colsWConflicts = []
        
        #step1 : check if current assignment is a solution
        for column in range(N):
            row = solution[column]    
            if (occupiedRows[row]>1) or (occupiedRightDiag[N-1-column-row]>1) or (occupiedLeftDiag[column-row]>1):
                colsWConflicts.append(column)
        if len(colsWConflicts) == 0 :
            return solution, num_steps
        
        #step2 : solution not found, so now we pick a random queen and begin picking the minimum conflict position. 
        var = np.random.choice(colsWConflicts)
        varRow = solution[var]
        
        #step3 : in order to move the queen we would remove it from it's current location. so adjust occupancy accordingly 
        occupiedRows[varRow] -= 1
        occupiedRightDiag[N-1-var-varRow] -= 1
        occupiedLeftDiag[var-varRow] -= 1 
        
        #step 4 : we want to find the value that has the minimum conflicts. apply same process as initialization
        minConflictRows = []
        minConflictCount = N 
        for row in range(N):
            realConflictCount = occupiedRows[row] + occupiedRightDiag[N-1-var-row] + occupiedLeftDiag[var-row]
            if realConflictCount == minConflictCount:
                minConflictRows.append(row)
            elif realConflictCount < minConflictCount:
                minConflictCount = realConflictCount
                minConflictRows = [row]
            else:
                pass
        
        #step 5: update solution assignment. use np.random.choice to break up ties. 
        value = np.random.choice(minConflictRows)
        solution[var] = value
        
        #step 6:update occupancy and conflict count. 
        occupiedRows[value] +=1 
        occupiedRightDiag[N-1-var-value] += 1
        occupiedLeftDiag[var-value] += 1
        
        #step 7: update step count. 
        num_steps += 1
        

    #return solution, num_steps


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 10
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
