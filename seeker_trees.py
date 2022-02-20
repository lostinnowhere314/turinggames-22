from cyberchase import Seeker, Hider, Match, Board
from typing import List, Tuple
import numpy as np
from scipy.special import softmax
import time

class Seeker_trees(Seeker):
    def __init__(self):
        self.MAX_SQUARES = 80
        self.PROB_CUTOFF = 1e-3
        
        self.impatience = 0
        self.MAX_IMPATIENCE = 2
        
        self.N_MOVES_KEPT = 18
        self.prev_moves = np.array([(2,2)] * self.N_MOVES_KEPT)
        self.PREV_WEIGHT = 12.
        self.PREV_DROPOFF = 2.0
        
        self.CAMP_TIMER = 260
        self.CAMP_TIMER_SEEK = 50
        
        self.hider_known = True
        self.hider_location = np.array((27,27))
        
        self.time_since_seen = np.zeros((30,30),dtype=int)
        
        self.passable = None
        self.hider_probabilities = None
        self.cum_hider_probabilities = None
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
        #Update which squares we can see
        self.time_since_seen += 1
        self.time_since_seen[visible_squares[:,0],visible_squares[:,1]] = 0
        
        # Keep track of turn number in case we want it
        self.i += 1
        
        self.hider_location = self._get_hider_loc(board_state, visible_squares)
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
            best_loc = self._get_best_target_loc()
        else:
            best_loc = self.hider_location
        
        if self.hider_location is not None and np.min(np.sum(np.abs(valid_moves - self.hider_location.reshape(1,-1)), axis=1)) < 0.5:
            # i.e. if we can move to the hider
            best_move = self.hider_location
        else:
            best_move = self._get_best_move(my_pos, best_loc, valid_moves)
        
        #update some things before returning
        self.prev_moves[self.i % self.N_MOVES_KEPT,:] = my_pos
        print(my_pos, best_move, self.hider_location)
        return tuple(best_move)
    
    def _get_best_target_loc(self):
        time_weights = 1 + np.sqrt(self.time_since_seen)
        
        if self.i < self.CAMP_TIMER:
            value = self.hider_probabilities * time_weights
        else:
            value = self.cum_hider_probabilities * time_weights
            
        mask = self.time_since_seen > self.CAMP_TIMER_SEEK
        if np.any(mask):
            time_weights[mask] += 0.05 * (self.time_since_seen[mask] - self.CAMP_TIMER_SEEK)**2
        
        best_loc_unraveled = np.argmax(value.ravel())
        return np.array(np.unravel_index(best_loc_unraveled, (30,30)))
    
    def init_hider_probabilities(self):
        self.hider_probabilities = np.zeros_like(self.passable, dtype=float)
        self.hider_probabilities[27, 27] = 1.
        self.cum_hider_probabilities = np.zeros_like(self.hider_probabilities)
        
    def reinit_hider_probabilities_random(self, n):
        self.hider_probabilities[:,:] = 0.
        for _ in range(n):
            x,y = np.random.randint(30,size=2)
            if self.passable[x,y]:
                self.hider_probabilities[x,y] = 1.
                break
        self.cum_hider_probabilities = np.copy(self.hider_probabilities)
    
    def update_hider_probabilities(self, board_state, visible_squares, max_squares):
        # Check if the hider is visible
        
        if self.hider_location is not None:
            #Reset probabilities and time-since-seen
            self.hider_known = True
            self.hider_probabilities[:,:] = 0.
            self.hider_probabilities[self.hider_location[0], self.hider_location[1]] = 1.
            self.time_since_seen[:,:] = 0
            self.cum_hider_probabilities = np.copy(self.hider_probabilities)
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
            
            #Update cum probabilities
            self.cum_hider_probabilities += self.hider_probabilities
            self.cum_hider_probabilities[visible_squares[:,0],visible_squares[:,1]] = 0.
        
    def _get_hider_loc(self, board_state, visible_squares) -> Tuple:
        """Vectorized version of function from scotland_yard.py.
        Returns None if hider is not in sight, otherwise returns hider's coordinate"""
        player2 = board_state[visible_squares[:,0],visible_squares[:,1]] == Board.PLAYER_2
        if np.any(player2):
            return visible_squares[np.argmax(player2)]
        else:
            return None
    
    def convert_id_to_cord(self, i):
        return i // 30, i % 30

    def convert_coord_to_id(self, y_location, x_location):
        return y_location * 30 + x_location

    def convert_to_index(self, i, j):
        return i * 30 + j

    def _get_best_move(self, start, end, dummy):

        unexplored_areas = {i for i in range(30*30)}


        # check if end points are valid
        if not(self.passable[start[0],start[1]]) or not(self.passable[end[0],end[1]]):
            return -1

        def check_if_valid(i, j):
            id = self.convert_to_index(i, j)
            if i > 0 and i < 30 and j > 0 and j < 30:
                if id in unexplored_areas and self.passable[i,j]:
                    return 1
            return 0

        possible_moves = [start]
        lengths_to_goal = [ np.linalg.norm(start - end) ]
        paths_used = [[start]]

        step_size = 1

        def look_around(i, j, paths_used_reserved):

            if i > 0:
                if check_if_valid(i - step_size, j):
                    possible_moves.append(np.array([i - step_size, j]))
                    lengths_to_goal.append( np.linalg.norm(np.array([i - step_size, j]) - end) )
                    paths_used.append( paths_used_reserved + [possible_moves[-1]] )
            if i < 30:
                if check_if_valid(i + step_size, j):
                    possible_moves.append(np.array([i + step_size, j]))
                    lengths_to_goal.append(np.linalg.norm(np.array([i + step_size, j]) - end))
                    paths_used.append( paths_used_reserved + [possible_moves[-1]] )
            if j > 0:
                if check_if_valid(i, j - step_size):
                    possible_moves.append(np.array([i, j - step_size]))
                    lengths_to_goal.append(np.linalg.norm(np.array([i, j - step_size]) - end))
                    paths_used.append( paths_used_reserved + [possible_moves[-1]] )
            if j < 30:
                if check_if_valid(i, j + step_size):
                    possible_moves.append(np.array([i, j + step_size]))
                    lengths_to_goal.append(np.linalg.norm(np.array([i, j + step_size]) - end))
                    paths_used.append( paths_used_reserved + [possible_moves[-1]] )

            # return possible_moves, lengths_to_goal

        iteration = 0

        start_time = time.time()

        while len(possible_moves) > 0:

            inverse_lengths = (1 / np.array(lengths_to_goal))**2
            probability_moves = softmax(inverse_lengths)

            if (time.time() - start_time) > 0.030:
                best_path = np.argmin(lengths_to_goal)
                if len(paths_used[best_path]) > 5:
                    return paths_used[best_path][4]
                return paths_used[best_path][-1]


            if np.isnan(probability_moves).any():
                best_path = np.where(probability_moves != probability_moves)[0][0]
                if len(paths_used[best_path]) > 5:
                    return paths_used[best_path][4]
                return paths_used[best_path][-1]

            path_to_explore = np.random.choice( len(probability_moves), 1, p=probability_moves )[0]

            paths_used_reserved = paths_used[path_to_explore]

            look_around(*possible_moves[path_to_explore], paths_used_reserved)

            try:
                unexplored_areas.remove(self.convert_to_index(*possible_moves[path_to_explore]))
            except:
                pass

            del possible_moves[path_to_explore]
            del lengths_to_goal[path_to_explore]
            del paths_used[path_to_explore]

            iteration += 1

