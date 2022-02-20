from cyberchase import Seeker, Hider, Match, Board
from typing import List, Tuple
import numpy as np

class Seeker2(Seeker):
    def __init__(self):
        self.MAX_SQUARES = 80
        self.PROB_CUTOFF = 1e-3
        
        self.impatience = 0.
        self.MAX_IMPATIENCE = 4.5
        self.MIN_IMPATIENCE = 2.5
        
        self.passable = None
        self.hider_probabilities = None
        self.i = -1
        self.hider_possible_moves = np.array([[ 0, -1,  0,  1, -2, -1,  0,  1,  2, -3, -2, -1,  0,  1,  2,  3,
            -4, -3, -2, -1,  0,  1,  2,  3,  4, -3, -2, -1,  0,  1,  2,  3,
            -2, -1,  0,  1,  2, -1,  0,  1,  0],
           [-4, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,
             2,  2,  2,  2,  2,  3,  3,  3,  4]])
    
    def get_action_from_state(self,
                board_state: List[List[int]],
                visible_squares: List[Tuple[int, int]],
                valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
                
        board_state = np.array(board_state)
        visible_squares = np.array(visible_squares)
        valid_moves = np.array(valid_moves)
        
        #Get our position from the board state
        my_pos = np.array(np.unravel_index(np.argmax(board_state.ravel() == Board.PLAYER_1), (30,30))).ravel()
        
        # Keep track of turn number in case we want it
        self.i += 1
        
        # Figure out where we want to go
        if self.passable is None:
            # Only runs on first turn
            self.passable = board_state != 3
            self.init_hider_probabilities()
        else:
            # Diffuse probabilities
            self.update_hider_probabilities(board_state, visible_squares, self.MAX_SQUARES) 
        
        #Find the best location, move towards it
        best_loc_unraveled = np.argmin(self.hider_probabilities.ravel())
        best_loc = np.array(np.unravel_index(best_loc_unraveled, (30,30)))
        #print(best_loc)
        
        best_move = self._get_best_move(my_pos, best_loc, valid_moves)
        
        return tuple(best_move)
        
    def init_hider_probabilities(self):
        self.hider_probabilities = np.zeros_like(self.passable, dtype=float)
        self.hider_probabilities[27, 27] = 1.
    
    def update_hider_probabilities(self, board_state, visible_squares, max_squares):
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
            pass_padded = np.full((38,38), False)
            pass_padded[4:-4,4:-4] = possible_squares
            
            #Initialize array
            new_probs = np.zeros((30, 30))
            
            # We can't do all of the squares because that takes too long
            update_mask = possible_squares * (self.hider_probabilities > self.PROB_CUTOFF)
            total_count = np.sum(update_mask)
            if total_count <= max_squares:
                #print(self.i,"Using all")
                coords = np.array(np.where(update_mask)).T
            else:
                #print(self.i,"Using sample")
                #Take only a subset of them
                coords_unraveled = np.random.choice(900, size=int(max_squares*1.1))
                coords = np.array(np.unravel_index(coords_unraveled, (30,30))).T
                mask = update_mask[coords[:,0], coords[:,1]]
                coords = coords[mask,:]
            
            #Update
            for x,y in coords:
                X = x+self.hider_possible_moves[0]
                Y = y+self.hider_possible_moves[1]
                #Check which are valid; this will exclude walls and also things that are out of bounds
                c_mask = pass_padded[X+4,Y+4]
                #k = np.sum(self.passable[X[c_mask],Y[c_mask]])
                k = np.sum(c_mask)
                new_probs[X[c_mask],Y[c_mask]] += self.hider_probabilities[x,y]/k
                
            # Preserve old probabilities if needed
            if total_count > max_squares:
                new_probs += self.hider_probabilities * (total_count / max_squares)
                
            # Normalize
            new_probs /= np.sum(new_probs)
            self.hider_probabilities = new_probs
        
    def _get_hider_loc(self, board_state, visible_squares) -> Tuple:
        """Vectorized version of function from scotland_yard.py.
        Returns None if hider is not in sight, otherwise returns hider's coordinate"""
        player2 = board_state[visible_squares[:,0],visible_squares[:,1]] == Board.PLAYER_2
        if np.any(player2):
            return visible_squares[np.argmax(player2)]
        else:
            return None
    
    def _get_best_move(self, my_pos, target, valid_moves):
        """
        valid_moves is an (n,2) ndarray
        """
        #Very naive, moves towards the closest square we want to go to
        
        chosen_move = None
        last = False
        deltas = np.sum(np.abs(valid_moves - my_pos), axis=1)
        option_distances = np.sum(np.abs(valid_moves - target) + np.random.random(valid_moves.shape)*1e-4,axis=1)
        
        if self.impatience > self.MAX_IMPATIENCE:
            self.impatience = self.MAX_IMPATIENCE
        
        while chosen_move is None:
            if self.impatience < self.MIN_IMPATIENCE:
                self.impatience = self.MIN_IMPATIENCE
                last = True
            mask = deltas > self.impatience
            if np.sum(mask)>0 or last:
                chosen_move = valid_moves[mask][np.argmax(option_distances[mask])]
                self.impatience = 5. - deltas[mask][np.argmax(option_distances[mask])]
            else:
                self.impatience -= 1.
        
        return chosen_move
    
