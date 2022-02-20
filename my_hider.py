import random
import numpy as np
from typing import List, Tuple
from cyberchase import Seeker, Hider, Match, Board

class My_hider(Hider):
    def __init__(self):
        self.pos = (27, 27)
        self.best_guesses = {(2, 2)}
        self.visible = False
        self.obj = None
        self.obj_camp_turns = 8
        self.quadrant_camp_turns = 20
        self.num_turns = 0
        self.new_quad = None
        
        self.passable = None
        
        #largest radius to consider for vision
        N = 27
        self.N_visioncalc = N
        #Things used in computing vision radii
        square_x = np.concatenate([
            np.linspace(-1,1,N+1),
            np.full(N-1, 1.),
            np.linspace(1,-1,N+1),
            np.full(N-1, -1.),
        ])
        square_y = np.concatenate([
            np.full(N, 1.),
            np.linspace(1,-1,N+1),
            np.full(N-1, -1.),
            np.linspace(-1,1,N, endpoint=False),
        ])
        square_coords = np.vstack((square_x, square_y))
        
        # (2,N,4N) array
        self.vision_vals = np.arange(1,N+1).reshape(1,-1,1) * square_coords.reshape(2,1,-1)
        self.indices = np.round(self.vision_vals).astype(int)

    def get_action_from_state(self, board_states: List[List[int]],
                              visible_squares: List[Tuple[int, int]],
                              valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        self.board_info = board_states
        self._check_visibility(board_states)

        if self.passable is None:
            # Only runs on first turn
            self.passable = np.array(self.board_info) != 3
            
        if self.obj is None:
            self.obj = visible_squares[np.argmax([np.linalg.norm(np.array(v) - np.array(self.pos), ord=1)
                                        for v in visible_squares])]
        if self.visible: # run away until invisible
            self.pos = self._get_visible_hider_move(valid_moves)
        elif self.num_turns < self.obj_camp_turns:
            s_radius_moves = self._get_radius(5)
            self.best_guesses = self._get_best_guesses(s_radius_moves, board_states)
            self.pos = self._travel_obj(valid_moves)
        elif self.num_turns % self.quadrant_camp_turns in range(12, 20):
            if random.random() < 0.5:
                self.pos = random.choice(valid_moves)
            else:
                self.pos = self._migrate(valid_moves)
            if self.num_turns % self.quadrant_camp_turns == 21:
                self.new_quad = None
            s_radius_moves = self._get_radius(5)
            self.best_guesses = self._get_best_guesses(s_radius_moves, board_states)
        else: # we are invisible run away from best guess of seeker locations
            s_radius_moves = self._get_radius(5)
            self.best_guesses = self._get_best_guesses(s_radius_moves, board_states)
            self.pos = self._get_hider_move(valid_moves)
        self.num_turns += 1
        return self.pos

    def _check_visibility(self, board_states):
        board_s = np.array(board_states)
        if 1 in board_s and self.num_turns > 3:
            # print("Visible")
            # print(board_s)
            self.visible = True
            self.best_guesses = {tuple(np.array(np.where(board_s == 1)).ravel())}
        else:
            self.visible = False


    def calculate_visible_tiles(self, loc):
        """
        Returns a (30,30) boolean ndarray of whether the tile is visible from that location
        """
        # Shift our index and find the valid parts
        loc = np.array(loc)
        index = self.indices + loc.reshape(2, 1, 1)
        index_mask = np.all((index >= 0) * (index < 30), axis=0)  # (N,4N)
        ind_x = index[0][index_mask]
        ind_y = index[1][index_mask]

        # Find passability and vision in radial-ness
        radial_passable = np.full_like(index_mask, False)  # (N,4N)
        radial_passable[index_mask] = self.passable[ind_x, ind_y]
        radial_visible = np.cumprod(radial_passable, axis=0)  # np.any(np.cumprod(radial_passable, axis=0), axis=2)

        # Convert back to map coords
        visible = np.full((30, 30), False)
        visible[index[0, :, :][index_mask[:, :]], index[1, :, :][index_mask[:, :]]] = radial_visible[index_mask[:, :]]

        return visible

    def _travel_obj(self, valid_moves):
        if self.obj is not None:
            move = valid_moves[np.argmin([np.linalg.norm(np.array(v) - np.array(self.obj), ord=1)
                                   for v in valid_moves])]
            return move
        move = valid_moves[np.argmin([np.linalg.norm(np.array(v) - np.array([15, 15]), ord=1) for v in valid_moves])]
        return move

    def _migrate(self, valid_moves):
        # quadrants as follows np.array([0, 1],[2, 3])
        if self.new_quad is None:
            self.new_quad = random.choice(range(4))
        if self.new_quad == 0:
            move = valid_moves[np.argmin([np.linalg.norm(v - np.array([7, 7]), ord=1)
                                         for v in valid_moves])]
        elif self.new_quad == 1:
            move = valid_moves[np.argmin([np.linalg.norm(v - np.array([7, 23]), ord=1)
                                         for v in valid_moves])]
        elif self.new_quad == 2:
            move = valid_moves[np.argmin([np.linalg.norm(v - np.array([23, 7]), ord=1)
                                         for v in valid_moves])]
        else: # quadrand 3
            move = valid_moves[np.argmin([np.linalg.norm(v - np.array([23, 23]), ord=1)
                                         for v in valid_moves])]
        return move

    def _get_radius(self, r):
        """
        Parameters -
            r (int) : desired radius, ie. the total number of steps the player can take
        """
        moves = []
        for bg in self.best_guesses:
            x = bg[0]
            y = bg[1]
            for i in range(r + 1):
                moves.append((x + i, y + (r - i)))
                moves.append((x + i, y - (r - i)))
                moves.append((x - i, y + (r - i)))
                moves.append((x - i, y - (r - i)))
        return set([m for m in moves if m[0] >= 0 and m[1] >= 0])

    def _get_best_guesses(self, radius_moves, board_states, num_guesses = 2):
        # hider_loc tuple (x, y)
        # radius_moves = set of tuples
        # valid_moves = set_of tuples
        # returns
        valid_moves = self.get_seeker_valid_moves(board_states)
        best_guesses = radius_moves.intersection(valid_moves)
        if len(best_guesses) < num_guesses:
            return best_guesses

        dist_tups = [(bg, np.linalg.norm(self.pos - bg, ord=1)) for bg in best_guesses]
        dist_tups.sort(key=lambda x: x[1])
        return set([b[0] for b in dist_tups[:num_guesses]])

    def _get_hider_move(self, valid_moves):
        move = valid_moves[np.argmax([sum([np.linalg.norm(np.array(v) - np.array(bg), ord=1)
                                          for bg in self.best_guesses])
                                     for v in valid_moves])]
        return move

    def _get_visible_hider_move(self, valid_moves):
        (seeker_loc,) = self.best_guesses
        s_visible_moves = self.calculate_visible_tiles(seeker_loc)
        # find which moves make the hider invisible to seeker
        h_valid_moves = [v for v in valid_moves if s_visible_moves[v[0], v[1]]]

        if len(h_valid_moves) > 0: # pick the spot that makes bot invisible and furthes away from seeker
            move = h_valid_moves[np.argmax([np.linalg.norm(np.array(v) - np.array(seeker_loc), ord=1)
                                    for v in h_valid_moves])]
            return move
        # No moves available that make your bot invisible
        return self._get_hider_move(valid_moves)

    def walk_board(self, start_pos, board_states, max_depth):
        # Perform BFS
        WALL = 3
        visited = np.zeros(board_states.shape)
        visited[start_pos] = 1
        queue = [(start_pos, 1)]
        while len(queue):
            curr, depth = queue.pop(0)
            neighbors = [(curr[0] + 1, curr[1]),
                         (curr[0] - 1, curr[1]),
                         (curr[0], curr[1] - 1),
                         (curr[0], curr[1] + 1)]
            for neighbor in neighbors:
                if not visited[neighbor] and board_states[neighbor] != WALL and depth < max_depth:
                    visited[neighbor] = 1
                    queue.append((neighbor, depth + 1))
        tuple_of_indices = np.where(visited == 1)
        return set([(i, j) for i, j in zip(tuple_of_indices[0], tuple_of_indices[1])])

    def get_seeker_valid_moves(self, board_states):
        player_movement = 5
        valid_moves = set()
        for bg in self.best_guesses:
            bg_valid_moves = set(self.walk_board(bg, board_states, player_movement))
            valid_moves.union(bg_valid_moves)
        return valid_moves