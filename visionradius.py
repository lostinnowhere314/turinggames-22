import random
from typing import List, Tuple
from cyberchase import Hider, Board
import numpy as np

class Visionradius(Hider):
    def __init__(self):
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
        

    def calculate_visible_tiles(self, loc):
        """
        Returns a (30,30) boolean ndarray of whether the tile is visible from that location
        """
        #Shift our index and find the valid parts
        index = self.indices + loc.reshape(2,1,1)
        index_mask = np.all((index >= 0) * (index < 30), axis=0) #(N,4N)
        ind_x = index[0][index_mask]
        ind_y = index[1][index_mask]
        
        #Find passability and vision in radial-ness
        radial_passable = np.full_like(index_mask, False) #(N,4N)
        radial_passable[index_mask] = self.passable[ind_x, ind_y]
        radial_visible = np.cumprod(radial_passable, axis=0) #np.any(np.cumprod(radial_passable, axis=0), axis=2)
        
        #Convert back to map coords
        visible = np.full((30,30), False)
        visible[index[0,:,:][index_mask[:,:]], index[1,:,:][index_mask[:,:]]] = radial_visible[index_mask[:,:]]
        
        return visible
        
        

    def get_action_from_state(self, board_state: List[List[int]],
                              visible_squares: List[Tuple[int, int]],
                              valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        
        board_state = np.array(board_state)
        visible_squares = np.array(visible_squares)
        valid_moves = np.array(valid_moves)
        
        my_pos = np.array(np.unravel_index(np.argmax((board_state.ravel() == Board.PLAYER_2)), (30,30))).ravel()
        
        if self.passable is None:
            self.passable = board_state != 3
        
        # Calculate visible tiles
        #visible = self.calculate_visible_tiles(my_pos)
        #vcount = np.sum(visible)
        #print(vcount, visible_squares.shape[0])
        
        move = random.choice(valid_moves)
        return tuple(move)
    
    def get_visible_squares(self, eye: Tuple[int, int], resolution=1):
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
    
    def visible_tiles_part2(self, eye):
        N = 30
        square_x = np.concatenate([
            np.arange(0,N),
            np.full(N-1, N-1),
            np.arange(0,N)[::-1],
            np.full(N-1, 0),
        ])
        square_y = np.concatenate([
            np.full(N, 0),
            np.linspace(1,-1,N+1),
            np.full(N-1, 0),
            np.arange(1,N)[::-1],
        ])
    