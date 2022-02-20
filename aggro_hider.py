from cyberchase import Seeker, Hider, Match, Board
import numpy as np
from typing import List, Tuple

#this tries to hide around a single block. Line-of-sight being funky makes this not work well
class Aggro_hider(Hider):
    def __init__(self):
        self.passable = None
        self.reached_target = False
        self.enemy_vision = np.full((30,30), False)
        self.i = -1
        
        self.N_MOVES_KEPT = 4
        self.prev_moves = np.array([(2,2)] * self.N_MOVES_KEPT)
        self.PREV_WEIGHT = 12.
        self.PREV_DROPOFF = 2.0
        
    def get_action_from_state(self, board_state: List[List[int]],
                              visible_squares: List[Tuple[int, int]],
                              valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        board_state = np.array(board_state)
        visible_squares = np.array(visible_squares)
        valid_moves = np.array(valid_moves)
        
        #Get our position from the board state
        my_pos = np.array(np.unravel_index(np.argmax((board_state.ravel() == Board.PLAYER_2)), (30,30))).ravel()
        
        # Keep track of turn number in case we want it
        self.i += 1
        
        if self.passable is None:
            # Only runs on first turn
            self.passable = board_state != 3
            
            # Figure out where we want to go and camp
            target_island_locs = self._find_isolated_pos()
            
            # Find which one is the closest
            if len(target_island_locs) > 0:
                index = np.argmin(np.sum(np.abs(target_island_locs - my_pos.reshape(1,-1)), axis=1))
                self.target = target_island_locs[index]
            else:
                self.target = None
        
        
        if self.target is None:
            chosen_move = valid_moves[0]
        else:
            if self.reached_target:
                # Figure out if we know where the seeker is, and if they can see us
                seeker_loc = self._get_seeker_loc(board_state, visible_squares)
                print(seeker_loc)
                if seeker_loc is None:
                    if np.sum(np.abs(my_pos - self.target)) <= 2:
                        chosen_move = my_pos
                    else:
                        chosen_move = self._get_best_move(my_pos, self.target, valid_moves)
                        #chosen_move = my_pos
                else:
                    # Determine if we are seen
                    seeker_vis_tiles = np.full((30,30), False)
                    vis_tiles = self.get_visible_squares(seeker_loc)
                    seeker_vis_tiles[vis_tiles[:,0],vis_tiles[:,1]] = True
                    if seeker_vis_tiles[my_pos[0], my_pos[1]]:
                        best_move = None
                        best_dist = 100.
                        # Out of the moves where the seeker can't see us, put us the closest to our target island
                        for move in valid_moves:
                            if not seeker_vis_tiles[move[0], move[1]]:
                                dist = np.sum(np.abs(move - self.target))
                                if dist < best_dist:
                                    best_dist = dist
                                    best_move = move
                        if best_move is not None:
                            chosen_move = best_move
                        #If we can't find such a move, then just move away from the seeker
                        else:
                            dists = np.sum(np.abs(valid_moves - seeker_loc.reshape(1,-1)),axis=1)
                            idx = np.argmax(dists)
                            chosen_move = valid_moves[idx]
                            
                    else:
                        chosen_move = my_pos
            else:
                # Move towards our target
                chosen_move = self._get_best_move(my_pos, self.target, valid_moves)
                if np.linalg.norm(my_pos - self.target) < 1.9:
                    self.reached_target = True
        
        
        self.prev_moves[self.i % self.N_MOVES_KEPT] = my_pos
        
        return tuple(chosen_move)

    def _get_seeker_loc(self, board_state, visible_squares) -> Tuple:
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
        #Very naive, moves towards the closest square we want to go to. But also repels itself
        
        deltas = np.sum(np.abs(valid_moves - my_pos), axis=1)
        option_distances = np.sum(np.abs(valid_moves - target) + np.random.random(valid_moves.shape)*1e-4,axis=1)
        chosen_move = valid_moves[np.argmin(option_distances)]
        
        return chosen_move

    def _find_isolated_pos(self):
        """Tries to find one of several small formations we can use to hide around"""
        valid = (
            self.passable[:-2,:-2] *
            self.passable[1:-1,:-2] *
            self.passable[2:,:-2] *
            self.passable[:-2,1:-1] *
            (~self.passable[1:-1,1:-1]) *
            self.passable[2:,1:-1] *
            self.passable[:-2,2:] *
            self.passable[1:-1,2:] *
            self.passable[2:,2:]
        )
        print(np.sum(valid))
        return np.array(np.where(valid)).T + 1
        #This part isn't working
        options = []
        options += self._find_mask(np.array([
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]))
        options += self._find_mask(np.array([
            [False, False, False, False],
            [False, True, True, False],
            [False, False, False, False],
        ]))
        options += self._find_mask(np.array([
            [False, False, False],
            [False, True, False],
            [False, True, False],
            [False, False, False],
        ]))
        options += self._find_mask(np.array([
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]))
        # can add more shapes
        return np.array(options) + 1
        
    def _find_mask(self, mask):
        """Mask is wall positions of the cluster"""
        w,h = mask.shape
        center_mask = np.full((30,30), False)
        center_mask[1:-w,1:-h] = True
        # Create an array of map pieces that match
        print('\t',np.sum(center_mask))
        for i in range(w):
            for j in range(h):
                vals = self.passable[i:-w+i,j:-h+j]
                if mask[i,j]:
                    vals = ~vals
                center_mask[i:-w+i,j:-h+j] *= vals
                print(i,j,'\t',np.sum(center_mask), mask[i,j], vals.shape, center_mask.shape)
        return list(np.array(np.where(center_mask)).T)
        
    def get_visible_squares(self, eye: Tuple[int, int], resolution=1):
        """This is the same computation as is done in board.py, but modified to use our passability matrix"""
        targets = []
        for i in range(0, self.width * resolution):
            targets.append((0, i / resolution))
            targets.append((29, i / resolution))
        for i in range(0, self.height * resolution):
            targets.append((i / resolution, 0))
            targets.append((i / resolution, 29))
        targets = list(set(targets))

        visible_squares = set([])

        for target in targets:
            dx = target[0] - eye[0]
            dy = target[1] - eye[1]
            nx = abs(dx)
            ny = abs(dy)
            sign_x = 1 if dx > 0 else -1
            sign_y = 1 if dy > 0 else -1

            p = list(eye)
            points = [tuple(p)]
            ix = 0
            iy = 0
            while ix < nx or iy < ny:
                decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
                if decision == 0:
                    # check if either corner is not a wall
                    corners = [list(p), list(p)]
                    corners[0][0] += sign_x
                    corners[1][1] += sign_y
                    if not (self.passable[corners[0][0], corners[0][1]] or
                            self.passable[corners[1][0], corners[1][1]]):
                        break
                    if self.passable[corners[0][0], corners[0][1]]:
                        visible_squares.add(tuple(corners[0]))
                    if self.passable[corners[1][0], corners[1][1]]:
                        visible_squares.add(tuple(corners[1]))
                    p[0] += sign_x
                    p[1] += sign_y
                    if not self.passable[p[0], p[1]]:
                        break
                    visible_squares.add(tuple(p))
                    ix += 1
                    iy += 1
                elif decision < 0:
                    p[0] += sign_x
                    if not self.passable[p[0], p[1]]:
                        break
                    visible_squares.add(tuple(p))
                    ix += 1
                else:
                    p[1] += sign_y
                    if not self.passable[p[0], p[1]]:
                        break
                    visible_squares.add(tuple(p))
                    iy += 1
        return np.array(visible_squares)