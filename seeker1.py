from cyberchase import Seeker, Hider, Match, Board
from typing import List, Tuple
import numpy as np

class Seeker1(Seeker):
    def __init__(self):
        self.passable = None
        self.hider_probabilities = None
        self.i = -1
        #self.hider_possible_moves = np.array([[ 0, -1,  0,  1, -2, -1,  0,  1,  2, -3, -2, -1,  0,  1,  2,  3,
            #-4, -3, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0,  1,  2,  3,
            #-2, -1,  0,  1,  2, -1,  0,  1,  0],
           #[-4, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,
             #0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,
             #2,  2,  2,  2,  2,  3,  3,  3,  4]])
        self.hider_possible_moves = np.array([[ 0, -1,  0,  1, -2, -1,  1,  2, -3, -2,  2,  3, -4, -3,  0,  3,
                     4, -3, -2,  2,  3, -2, -1,  1,  2, -1,  0,  1,  0],
                   [-4, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1,  0,  0,  0,  0,
                     0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  4]])
    
    def get_action_from_state(self,
                board_state: List[List[int]],
                visible_squares: List[Tuple[int, int]],
                valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
                
        board_state = np.array(board_state)
        visible_squares = np.array(visible_squares)
        valid_moves = np.array(valid_moves)
        
        # Keep track of turn number in case we want it
        self.i += 1
        
        # Figure out where we want to go
        if self.passable is None:
            # Only runs on first turn
            self.passable = board_state != 3
            self.init_hider_probabilities()
        else:
            # Diffuse probabilities
            self.update_hider_probabilities(board_state, visible_squares) 
        
        #Find the best location, move towards it
        best_loc_unraveled = np.argmax(self.hider_probabilities.ravel())
        best_loc = np.array(np.unravel_index(best_loc_unraveled, (30,30)))
        
        best_move = self._get_best_move(best_loc, valid_moves)
        
        return tuple(best_move)
        
    def init_hider_probabilities(self):
        self.hider_probabilities = np.zeros_like(self.passable, dtype=float)
        self.hider_probabilities[27, 27] = 1.
    
    def update_hider_probabilities(self, board_state, visible_squares):
        # Check if the hider is visible
        hider_loc = self._get_hider_loc(board_state, visible_squares)
        if hider_loc is not None:
            self.hider_probabilities[:,:] = 0.
            self.hider_probabilities[hider_loc[0], hider_loc[1]] = 1.
        else:
            # Otherwise, update probabilities
            # Find all possibly valid squares
            possible_squares = np.copy(self.passable)
            possible_squares[visible_squares[:,0],visible_squares[:,1]] = False
            
            #Update probabilities
            probs_padded = np.zeros((38,38))
            probs_padded[4:-4,4:-4] = self.hider_probabilities
            
            #pass_padded = np.full((38,38), False)
            #pass_padded[4:-4,4:-4] = possible_squares
            
            self.hider_probabilities[:,:] = 0.
            coords = np.array(np.where(possible_squares)).T
            for x,y in coords:
                X = x+4+self.hider_possible_moves[0]
                Y = y+4+self.hider_possible_moves[1]
                self.hider_probabilities[x,y] = np.sum(probs_padded[X,Y])
            
            # Normalize
            self.hider_probabilities /= np.sum(self.hider_probabilities)
        
    def _get_hider_loc(self, board_state, visible_squares) -> Tuple:
        """Vectorized version of function from scotland_yard.py.
        Returns None if hider is not in sight, otherwise returns hider's coordinate"""
        player2 = board_state[visible_squares[:,0],visible_squares[:,1]] == Board.PLAYER_2
        if np.any(player2):
            return visible_squares[np.argmax(player2)]
        else:
            return None
    
    def _get_best_move(self, target, valid_moves):
        #Very naive, moves towards the closest square we want to go to
        option_distances = np.sum(np.abs(valid_moves - target) + np.random.random(valid_moves.shape)*1e-4,axis=1)
        return valid_moves[np.argmax(option_distances)]
    
