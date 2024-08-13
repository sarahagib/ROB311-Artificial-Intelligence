from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        #print([move1, move2])
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)
        
        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]
        #print(p1_score, p2_score)

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    my student agent will be a markov-ish implementation with memory. 
    the goal is to keep track of the opponent's last moves in order to predict
    it's next move given a particular state by adjusting the transition matrix 
    probabilities after each turn. 
    
    recall: rock = 0, paper = 1, scissor = 2. 
    
    at any given point, the game is at 1 of 9 states where the first number is 
    my move and the second number is the opponent's move. When initializing the 
    game each state has equal count (1 - to avoid division by 0 errors) and 
    equal probability (1/3). 
    states:             count:           probabilities:
    [[00, 01, 02],      [[1, 1, 1],      [[1/3, 1/3, 1/3],
     [10, 11, 12],       [1, 1, 1],       [1/3, 1/3, 1/3],
     [20, 21, 22]]       [1, 1, 1]]       [1/3, 1/3, 1/3]]
    
    I have chosen to represent any given state of the game and it's respective
    state information using a dictionary where the key is my move (0, 1, 2) and 
    the value is a 3x3 in which each row represent the opponent's move and it 
    contains either the updated probabilities given a specific state or the 
    updated count of the opponent's past moves.(transition_matrix & moves_counter). 
    
    Here is the approach: when the game starts, we have no history so my agent 
    chooses a random move. After the first iteration and for all subsequent 
    interations, my_last move and the opp_last move are remembered. For each 
    iteration, we attempt to detect if the opposing player is either the first 
    move player, the copycat player or the goldfish player as these are the 
    'simple' opponents that do not follow any type of probabilistics algorithm. 
    
    the first move detector is a counter that is incremented every time the 
    opponent's current and past move is rock, and the counter is set to 0 when a 
    move that isn't rock is played. 
    
    the copycat detector is a counter that is incremented every time the opponent
    current move is equal to my past move. It is impossible for a copycat player 
    to play the same move the goldfish opponent would play, so the goldfish 
    counter is reset to 0. 
    
    the goldfish detector is a counter that is incremented every time the opponent 
    current move is the move that would have beat my past move. It is impossible 
    for the goldfish player to play the same move the copycat would play, so the
    copycat counter is reset to 0 here. 
    
    if any of these counters reach 10, as in, 10 sequential moves were played 
    following one of the specific agent patterns, then my next move is chosen 
    to beat that opponent. 
    
    If none of these counters reach 10, we update the moves_counter using the 
    state information, and then update the transition matrix probabilities by 
    dividing the count of each individual move done by the opponent given the 
    past state by the sum of all moves done given that previous state. 
    
    For example 
    -> my_past = 0, opp_past = 1, my_curr = 2, opp_curr = 0 
    
    moves_counter[0][1,0] += 1
    past_moves_tot = np.sum(moves_counter[0][1])
    for i in range(3):
        transition_matrix[0][1,i] = (moves_counter[0][1,i] / past_move_sum)
    
    after updating the above information, we index into the transition_matrix 
    dictionary given the current state (as in the moves that we just played and
    have not become past moves just yet) and guess that the opponent will pick 
    the move with the highest probability given the current state. Given our 
    guess of what the opponent will play, we then chose the move that will beat 
    them and set that as our next move and then update the respective last moves
    with the moves of this game. 
    
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        # YOUR CODE GOES HERE
        
        #REMEMBER:
            # rock = 0 
            # paper = 1 
            # scissors = 2
        
        self.move = np.random.randint(0, self.n_moves)
        self.my_past = None
        self.opp_past = None
        
        self.goldfish_detector = 0 
        self.copycat_detector = 0 
        self.firstmove_detector = 0 
        
        #if opponent picks rock we pick paper, 
        #if they pick paper we pick scissors, 
        #if they pick scissors we pick rock
        self.choose_next_move = {0:1, 1:2, 2:0}
        
        #if my past move was 0, then goldfish will play 1 
        #if my past move was 1, then goldfish will play 2 
        #if my past move was 2, then goldfish will play 3
        self.goldfish_moves = {0:1, 1:2, 2:0}
        
        
        
        #key = our move 
        #opp_last_move = the first index into the respective transitions matrix. 
        #goal is to update the probabilities that estimate the opponent's next move 
        self.transition_matrix = {0: np.array([[1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3]]), 
                                  1: np.array([[1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3]]), 
                                  2: np.array([[1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3]])
                                  }
        # find location based on our and opp previous move then increment on 
        # the move that the opp did, to keep track. 
        # serves as memory. 
        self.moves_counter = {0: np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]), 
                              1: np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]), 
                              2: np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]])
                              }
        pass

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # YOUR CODE GOES HERE
        return self.move 
        pass

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE
        
        #step 1: update the information
        if self.my_past == None:
            #runs on first iteration. 
            self.move = np.random.randint(0, self.n_moves) 
            
        else:
            if other_move == 0 and self.opp_past == 0:
                self.firstmove_detector += 1
            if other_move == self.my_past and self.my_past != 0 :
                self.copycat_detector += 1 
                self.goldfish_detector = 0 
                self.firstmove_detector = 0 
            if other_move == self.my_past and self.my_past == 0 : 
                self.copycat_detector += 1 
                self.goldfish_detector = 0 
            if other_move == self.choose_next_move[self.my_past] :
                self.goldfish_detector += 1
                self.copycat_detector = 0 
                self.firstmove_detector = 0 
            
            if self.firstmove_detector >= 10 : 
                self.move = 1 
                
            elif self.copycat_detector >= 10 : 
                self.move = self.choose_next_move[self.my_past]
                
            elif self.goldfish_detector >= 10: 
                potential_opponent_move = self.goldfish_moves[self.my_past]
                self.move = self.choose_next_move[potential_opponent_move]
                
            else: 
                #runs on all subsequent iterations
                self.moves_counter[self.my_past][self.opp_past, other_move] += 1 
                past_moves_tot = np.sum(self.moves_counter[self.my_past][self.opp_past])
                for i in range(0, self.n_moves):
                    self.transition_matrix[self.my_past][self.opp_past, i] = (self.moves_counter[self.my_past][self.opp_past,i] / past_moves_tot)
                #step 2: pick the next move
                potential_opponent_move = np.argmax(self.transition_matrix[my_move][other_move])
                self.move = self.choose_next_move[potential_opponent_move]
        
        self.my_past = my_move 
        self.opp_past = other_move 
        
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.move = np.random.randint(0, self.n_moves)
        self.my_past = None
        self.opp_past = None
        
        #reset transition matrix to orginal probabilities
        self.transition_matrix = {0: np.array([[1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3]]), 
                                  1: np.array([[1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3]]), 
                                  2: np.array([[1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3],
                                               [1/3, 1/3, 1/3]])
                                  }
        #reset memory to no games played. 
        self.moves_counter = {0: np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]), 
                              1: np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]]), 
                              2: np.array([[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]])
                              }
        pass


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    copy_cat_player = CopycatPlayer(game_matrix)
    
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))
    
    student_score, uniform_score = play_game(student_player, uniform_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("Uniform player's score: {:}".format(uniform_score))
    
    student_score, copy_cat_score = play_game(student_player, copy_cat_player, game_matrix)
    
    print("Your player's score: {:}".format(student_score))
    print("CopyCat player's score: {:}".format(copy_cat_score))
    