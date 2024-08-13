import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    
    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    ### YOUR CODE GOES HERE
    
    #start by mapping out the board for each row and diagoanal orientation. 
    #value indicates if position is occupied. 
    #position is occupied iff a queen is in it, or iff the position is diagonal to a placed queen. 
    
    rows = dict()
    for i in range(0,N):
        rows[i] = 0
    
    right_diag = dict()
    for i in range(-N+1, N):
        #think of it as drawing diagonal lines from the rightmost corner to the left. 
        #0 = the corner. -1 to -3 from the downwards, 1 to 3 to the left. 
        right_diag[i] = 0
    
    left_diag = dict()
    for i in range(-N+1, N):
        #think of it as drawing diagonal lines from the leftmost corner to the right. 
        #0 = the corner, -1 to -3 going downwards, 1 to 3 going to the right
        left_diag[i] = 0
    
    #fill in the occupied/conflicted locations 
    rows[greedy_init[0]] = 1
    right_diag[((N-1)-greedy_init[0])] = 1
    left_diag[(-greedy_init[0])] = 1
    
    
    for column in range(1, N):
        potentialConflicts = N #might change to 3 because technically can only have up down, right conflicts. 
        potentialMoves = []
        
        #go through every row in one column and try to find the move with the least amount of conflicts. 
        for row in range(0, N):
            
            #get the number that identifies the diagonals that intesect here
            rdiag = N-1-column-row
            ldiag = column-row
            
            #collect the number of surrounding conflicts based on dictionaries 
            realConflicts = rows[row] + right_diag[rdiag] + left_diag[ldiag]
            
            #try to pick the best move
            if realConflicts == potentialConflicts:
                #if there is an equal amount of conflict in the move, just append and randomly pick later. 
                potentialMoves.append(row)
            elif realConflicts < potentialConflicts :
                #if there is a move that has less conflict than the rest, reset the potential moves to only that option. 
                potentialConflicts = realConflicts
                potentialMoves = [row]
            else:
                pass
        
        #after going through every row in one column, randomly pick a move
        #place it in the greedy_init at the correct column
        move = np.random.choice(potentialMoves)
        greedy_init[column] = move 
        
        #update the dictionaries to count the occupied and conflicting spaces. 
        occupiedRow = greedy_init[column]
        rows[occupiedRow] +=1
        
        rdiag = N-1-column - occupiedRow
        right_diag[rdiag] += 1
        
        ldiag = column - occupiedRow 
        left_diag[ldiag] +=1
    
    
    return greedy_init


if __name__ == '__main__':
    init_queens = initialize_greedy_n_queens(4)
    print(init_queens)
    pass
