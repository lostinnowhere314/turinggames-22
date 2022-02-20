from cyberchase import Seeker, Hider, Match, Board
from typing import List, Tuple
import numpy as np

class Seeker_selfavoid(Seeker):
    def __init__(self):
        print("Hello there")
        self.MAX_SQUARES = 80
        self.PROB_CUTOFF = 1e-3
        
        self.impatience = 0
        self.MAX_IMPATIENCE = 2
        
        self.N_MOVES_KEPT = 15
        self.prev_moves = np.array([(2,2)] * self.N_MOVES_KEPT)
        self.PREV_WEIGHT = 10.
        self.PREV_DROPOFF = 2.0
        
        self.hider_known = True
        self.hider_location = np.array((27,27))
        
        self.passable = None
        self.hider_probabilities = None
        #self.passable_nonvisible = None
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
        my_pos = np.array(np.unravel_index(np.argmax((board_state.ravel() == Board.PLAYER_1)), (30,30))).ravel()
        
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
        if self.hider_location is None:
            best_loc_unraveled = np.argmax(self.hider_probabilities.ravel())
            best_loc = np.array(np.unravel_index(best_loc_unraveled, (30,30)))
        else:
            best_loc = self.hider_location
        
        best_move = self._get_best_move(my_pos, best_loc, valid_moves)
        
        self.prev_moves[self.i % self.N_MOVES_KEPT,:] = my_pos
        
        return tuple(best_move)
        
    def init_hider_probabilities(self):
        self.hider_probabilities = np.zeros_like(self.passable, dtype=float)
        self.hider_probabilities[27, 27] = 1.
        
    def reinit_hider_probabilities_random(self, n):
        self.hider_probabilities[:,:] = 0.
        for _ in range(n):
            x,y = np.random.randint(30,size=2)
            if self.passable[x,y]:
                self.hider_probabilities[x,y] = 1.
                break
    
    def update_hider_probabilities(self, board_state, visible_squares, max_squares):
        # Check if the hider is visible
        self.hider_location = self._get_hider_loc(board_state, visible_squares)
        if self.hider_location is not None:
            self.hider_known = True
            self.hider_probabilities[:,:] = 0.
            self.hider_probabilities[self.hider_location[0], self.hider_location[1]] = 1.
        else:
            self.hider_known = False
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
                new_probs += self.hider_probabilities * (total_count / max_squares) * update_mask
                
            if np.sum(new_probs) > 0:
                # Normalize
                new_probs /= np.sum(new_probs)
                self.hider_probabilities = new_probs
            else:
                #Something went wrong
                self.reinit_hider_probabilities_random(3)
        
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
        
        last = False
        deltas = np.sum(np.abs(valid_moves - my_pos), axis=1)
        option_distances = np.sum(np.abs(valid_moves - target) + np.random.random(valid_moves.shape)*1e-4,axis=1)
        #The matrix constructed is (n,m,2) where n is # valid moves, m is # previous kept moves
        # Weight against staying in the same place
        if self.hider_known:
            #print("Hider location known")
            dist_weights = np.zeros_like(option_distances)
        else:
            own_distances = np.sum(np.abs(valid_moves.reshape(-1,1,2) - self.prev_moves.reshape(1,-1,2)), axis=2)
            dist_weights = np.sum(self.PREV_WEIGHT/(1+self.PREV_DROPOFF*own_distances), axis=1)
        
        chosen_move = valid_moves[np.argmin(option_distances + dist_weights)]
        
        #if self.hider_known:
        #    print(my_pos, chosen_move, self.hider_location)
        return chosen_move
        
    
