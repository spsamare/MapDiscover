import cv2 as cv
import numpy as np
from numpy.random import randint
from tqdm import tqdm

DOT_GAP = 2
BLOCK_WIDTH = 20
LINE_WIDTH = 4  # 4
NUM_BLOCKS = 10
MAP_WIDTH = DOT_GAP * BLOCK_WIDTH * (NUM_BLOCKS + 1)
MAP_X = [*range(0, MAP_WIDTH, DOT_GAP * BLOCK_WIDTH)]
MAP_Y = [*range(0, MAP_WIDTH, DOT_GAP * BLOCK_WIDTH)]
SHOW_MAP = True
SHIFT_GAP = 5  # 5
VIEW_H = 20

np.random.seed(0)
MAT_A = np.random.rand(4, 4) - .5
MAT_B = np.random.rand(4) - .5

WRAP_AROUND = False
MAP_ID = 0


def id_to_index(id_):
    id_mod = id_ % NUM_BLOCKS
    return NUM_BLOCKS * id_mod[0] + id_mod[1]


def index_to_id(ind):
    ind = ind % (NUM_BLOCKS * NUM_BLOCKS)
    return ind // NUM_BLOCKS, ind % NUM_BLOCKS


def show(img, height=2*VIEW_H*(NUM_BLOCKS+1), title='untitled'):
    if SHOW_MAP:
        # height = 241
        width = int(img.shape[1] * height / img.shape[0])
        dim = (width, height)

        # resize image
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        # draw grid
        val = 0
        for i in range(NUM_BLOCKS + 1):
            resized[2*VIEW_H*i + VIEW_H+1, :] = val
            resized[:, 2 * VIEW_H * i + VIEW_H + 1] = val

        cv.namedWindow(title)  # Create a named window
        cv.moveWindow(title, 40, 30)  # Move it to (40,30)
        cv.imshow(title, resized)
        cv.waitKey()
        cv.destroyAllWindows()


def update_connectivity(source_node, connectivity, branch=None):
    if branch is None:
        branch = [0, 1, 2, 3]
    else:
        branch = [branch]
    node_s = id_to_index(source_node.id)
    for _ in branch:
        node_d = source_node.neighbours_index[_]
        connectivity[node_s, node_d] = source_node.state[_]


def is_disconnected(connectivity):
    u_tri_sum = np.sum(np.triu(connectivity), axis=0)
    return np.any(u_tri_sum[1:] == 0)


class Dotter:
    def __init__(self, big_map, origin, is_wrapped=True):
        self.id = origin
        self.xy = origin + [1, 1]
        self.active = False
        self.state = np.array([0, 0, 0, 0])
        self.neighbours = [
            self.id - [0, 1],
            self.id - [1, 0],
            (self.id + [0, 1]) % NUM_BLOCKS,
            (self.id + [1, 0]) % NUM_BLOCKS
        ]
        self.neighbours_index = [id_to_index(_) for _ in self.neighbours]
        self.pointers = None
        self.map = big_map
        self.connectivity = None
        self.is_wrapped = is_wrapped
        if not is_wrapped:
            if np.any(self.id - [0, 1] < 0):
                self.neighbours[0] = None
            if np.any(self.id - [1, 0] < 0):
                self.neighbours[1] = None
            if np.any(self.id + [0, 1] >= NUM_BLOCKS):
                self.neighbours[2] = None
            if np.any(self.id + [1, 0] >= NUM_BLOCKS):
                self.neighbours[3] = None

    def init(self, pointers, connectivity):
        self.pointers = pointers
        self.active = True
        if self.is_wrapped:
            self.state = np.array([0, 0, 0, 0]) + 1
        else:
            for _ in range(4):
                if self.neighbours[_] is None:
                    self.state[_] = 0
                else:
                    self.state[_] = 1
        self.connectivity = connectivity
        update_connectivity(self, connectivity)

    def evolve(self):
        # self.update_branch(branch_id=3)
        if self.active:
            n_active = np.array([0, 0, 0, 0])
            n_state = np.zeros((4, 4))
            dynamic_branch = np.ones(4)
            for _ in range(4):
                n_id = self.neighbours[_]
                if n_id is not None:
                    n_active[_] = self.pointers[n_id[0]][n_id[1]].active
                    n_state[_, :] = self.pointers[n_id[0]][n_id[1]].state
                else:
                    n_active[_] = False
                    n_state[_, :] = np.zeros(4)
                    dynamic_branch[_] = 0
            indict = (np.matmul(np.matmul(np.matmul(np.transpose(n_state), MAT_A), n_state), MAT_B) >
                      self.state * MAT_B) * dynamic_branch
            self.active = False
            for _ in range(4):
                if indict[_]:
                    self.update_branch(_)
                    if is_disconnected(self.connectivity):
                        self.update_branch(_)
                        # print(is_disconnected(self.connectivity))
                    else:
                        self.active = True

    def update_branch(self, branch_id):
        # Alter the state
        self.state[branch_id] = 1 - self.state[branch_id]
        # Modify the neighbour
        n_id = self.neighbours[branch_id]
        self.pointers[n_id[0]][n_id[1]].state[(branch_id + 2) % 4] = self.state[branch_id]
        #
        update_connectivity(self, self.connectivity, branch=branch_id)
        update_connectivity(self.pointers[n_id[0]][n_id[1]], self.connectivity)

    def update_map(self):
        #
        self.map[
        (MAP_X[self.xy[0]] - LINE_WIDTH):(MAP_X[self.xy[0]] + LINE_WIDTH + 1),
        np.maximum(0, MAP_Y[self.xy[1]] + 1 - DOT_GAP * BLOCK_WIDTH + LINE_WIDTH):(MAP_Y[self.xy[1]] - LINE_WIDTH)
        ] = 255 * (1 - self.state[0])
        #
        self.map[
        np.maximum(0, MAP_X[self.xy[0]] + 1 - DOT_GAP * BLOCK_WIDTH + LINE_WIDTH):(MAP_X[self.xy[0]] - LINE_WIDTH),
        (MAP_Y[self.xy[1]] - LINE_WIDTH):(MAP_Y[self.xy[1]] + LINE_WIDTH + 1)
        ] = 255 * (1 - self.state[1])
        #
        self.map[
        (MAP_X[self.xy[0]] - LINE_WIDTH):(MAP_X[self.xy[0]] + LINE_WIDTH + 1),
        (MAP_Y[self.xy[1]] + LINE_WIDTH + 1):(MAP_Y[self.xy[1]] + DOT_GAP * BLOCK_WIDTH - LINE_WIDTH + 1),
        ] = 255 * (1 - self.state[2])
        #
        self.map[
        (MAP_X[self.xy[0]] + LINE_WIDTH + 1):(MAP_X[self.xy[0]] + DOT_GAP * BLOCK_WIDTH - LINE_WIDTH + 1),
        (MAP_Y[self.xy[1]] - LINE_WIDTH):(MAP_Y[self.xy[1]] + LINE_WIDTH + 1)
        ] = 255 * (1 - self.state[3])
        """"""
        self.map[
        (MAP_X[self.xy[0]] - LINE_WIDTH):(MAP_X[self.xy[0]] + LINE_WIDTH + 1),
        (MAP_Y[self.xy[1]] - LINE_WIDTH):(MAP_Y[self.xy[1]] + LINE_WIDTH + 1)
        ] = 255 * (1 - np.minimum(np.sum(self.state), 1))
        """"""


def generate(map_=None):
    if map_ is None:
        map_ = 'map-grid'
    my_map = 255 * np.ones((MAP_WIDTH, MAP_WIDTH), dtype=np.float32)
    #
    all_dots = []
    global_connectivity = np.zeros((NUM_BLOCKS ** 2, NUM_BLOCKS ** 2))
    # dot generation
    for i in range(NUM_BLOCKS):
        row_dots = []
        for j in range(NUM_BLOCKS):
            row_dots.append(
                Dotter(big_map=my_map, origin=np.array([i, j]), is_wrapped=WRAP_AROUND)
            )
        all_dots.append(row_dots)

    # dot initialize
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            all_dots[i][j].init(pointers=all_dots, connectivity=global_connectivity)
    # print(np.sum(global_connectivity, 0))
    print('Is path disconnected:', is_disconnected(global_connectivity))
    # test generation
    running = True
    # old_activity = np.ones(NUM_BLOCKS*NUM_BLOCKS)
    old_count = NUM_BLOCKS * NUM_BLOCKS
    end_counter = 0
    while running:
        # running = False
        # now_active = np.zeros(NUM_BLOCKS*NUM_BLOCKS)
        active_count = 0
        for k in range(NUM_BLOCKS*NUM_BLOCKS):
            i, j = index_to_id(k+MAP_ID)
            all_dots[i][j].evolve()
            active_count = active_count + all_dots[i][j].active
        """
        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                all_dots[i][j].evolve()
                # running = running or all_dots[i][j].active
                active_count = active_count + all_dots[i][j].active
                # now_active[id_to_index(all_dots[i][j].id)] = all_dots[i][j].active
        """
        # print(active_count)
        if active_count == old_count:
            end_counter += 1
        else:
            end_counter = 0
        running = end_counter < 7
        old_count = active_count
        # running = np.any(old_activity != now_active)
        # if active_count < 1:  # .3 * NUM_BLOCKS ** 2:
        #     running = False
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            all_dots[i][j].update_map()
    # print(np.sum(global_connectivity, 0))
    print('Is path disconnected:', is_disconnected(global_connectivity))
    show(img=my_map)
    # cv.imwrite('res/' + map_ + '.jpg', my_map)


if __name__ == '__main__':
    for map_id in range(1):
        generate(map_=str(map_id))
        # img_ = 'test/image005'
        # deform(map_=str(map_id))
