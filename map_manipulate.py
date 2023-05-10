import cv2 as cv
import numpy as np
from numpy.random import randint
from tqdm import tqdm

DOT_GAP = 2
BLOCK_WIDTH = 20
LINE_WIDTH = 10  # 4
NUM_BLOCKS = 10
MAP_WIDTH = DOT_GAP * BLOCK_WIDTH * (NUM_BLOCKS + 1)
BRANCH_PROB = .5
MAP_X = [*range(0, MAP_WIDTH, DOT_GAP * BLOCK_WIDTH)]
MAP_Y = [*range(0, MAP_WIDTH, DOT_GAP * BLOCK_WIDTH)]
SHOW_MAP = False
SHIFT_GAP = 5


def show(img, height=300, title='untitled'):
    if SHOW_MAP:
        # height = 241
        width = int(img.shape[1] * height / img.shape[0])
        dim = (width, height)

        # resize image
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        cv.namedWindow(title)  # Create a named window
        cv.moveWindow(title, 40, 30)  # Move it to (40,30)
        cv.imshow(title, resized)
        cv.waitKey()
        cv.destroyAllWindows()


class Dotter:
    def __init__(self, big_map, big_occupancy, dot_list, origin, tail=-1):
        self.xy = origin
        self.branches = [0, 1, 2, 3]
        p = .5  # 1 - (np.sum(big_occupancy) / (NUM_BLOCKS + 2) ** 2) ** 2
        self.prob = p * np.ones(4)
        self.place = [1, 1, 1, 1]
        if tail >= 0:
            self.prob[tail] = 0
        else:
            self.prob[randint(0, 4)] = 1.
        # further pruning
        if big_occupancy[origin[0] + 1, origin[1]] == 1:
            self.place[0] = 0
        if big_occupancy[origin[0] - 1, origin[1]] == 1:
            self.place[3] = 0
        if big_occupancy[origin[0], origin[1] + 1] == 1:
            self.place[1] = 0
        if big_occupancy[origin[0], origin[1] - 1] == 1:
            self.place[2] = 0
        #
        self.dot_list = dot_list
        self.map = big_map
        self.occupancy = big_occupancy
        self.occupancy[origin[0], origin[1]] = 1

    def grow(self):
        will_grow = np.random.rand(4) <= self.prob
        for _ in range(0, len(self.branches)):
            if will_grow[_]:
                if self.branches[_] == 0:
                    self.map[
                    (MAP_X[self.xy[0]] - LINE_WIDTH):(MAP_X[self.xy[0]] + DOT_GAP * BLOCK_WIDTH + LINE_WIDTH + 1),
                    (MAP_Y[self.xy[1]] - LINE_WIDTH):(MAP_Y[self.xy[1]] + LINE_WIDTH + 1)
                    ] = 0
                    if self.place[_] == 1:
                        self.dot_list.append(Dotter(big_map=self.map, big_occupancy=self.occupancy,
                                                    dot_list=self.dot_list,
                                                    origin=(self.xy[0] + 1, self.xy[1]), tail=3))
                elif self.branches[_] == 3:
                    self.map[
                    np.maximum(0, MAP_X[self.xy[0]] + 1 - DOT_GAP * BLOCK_WIDTH - LINE_WIDTH):
                    (MAP_X[self.xy[0]] + 1 + LINE_WIDTH),
                    (MAP_Y[self.xy[1]] - LINE_WIDTH):(MAP_Y[self.xy[1]] + LINE_WIDTH + 1)
                    ] = 0
                    if self.place[_] == 1:
                        self.dot_list.append(Dotter(big_map=self.map, big_occupancy=self.occupancy,
                                                    dot_list=self.dot_list,
                                                    origin=(self.xy[0] - 1, self.xy[1]), tail=0))
                elif self.branches[_] == 1:
                    self.map[
                    (MAP_X[self.xy[0]] - LINE_WIDTH):(MAP_X[self.xy[0]] + LINE_WIDTH + 1),
                    (MAP_Y[self.xy[1]] - LINE_WIDTH):(MAP_Y[self.xy[1]] + DOT_GAP * BLOCK_WIDTH + LINE_WIDTH + 1),
                    ] = 0
                    if self.place[_] == 1:
                        self.dot_list.append(Dotter(big_map=self.map, big_occupancy=self.occupancy,
                                                    dot_list=self.dot_list,
                                                    origin=(self.xy[0], self.xy[1] + 1), tail=2))
                elif self.branches[_] == 2:
                    self.map[
                    (MAP_X[self.xy[0]] - LINE_WIDTH):(MAP_X[self.xy[0]] + LINE_WIDTH + 1),
                    np.maximum(0, MAP_Y[self.xy[1]] + 1 - DOT_GAP * BLOCK_WIDTH - LINE_WIDTH):
                    (MAP_Y[self.xy[1]] + 1 + LINE_WIDTH)
                    ] = 0
                    if self.place[_] == 1:
                        self.dot_list.append(Dotter(big_map=self.map, big_occupancy=self.occupancy,
                                                    dot_list=self.dot_list,
                                                    origin=(self.xy[0], self.xy[1] - 1), tail=1))
        self.dot_list.remove(self)


def generate():
    my_map = 255 * np.ones((MAP_WIDTH + 1, MAP_WIDTH + 1), dtype=np.float32)
    occupancy = np.zeros((NUM_BLOCKS + 2, NUM_BLOCKS + 2))
    occupancy[0, :] = 1
    occupancy[-1, :] = 1
    occupancy[:, 0] = 1
    occupancy[:, -1] = 1
    # print(occupancy)
    #
    all_dots = []
    # initial dot generation
    location = [NUM_BLOCKS // 2 + 1, NUM_BLOCKS // 2 + 1]  # np.random.randint(1, NUM_BLOCKS, 2)
    all_dots.append(
        Dotter(big_map=my_map, big_occupancy=occupancy, dot_list=all_dots, origin=(location[0], location[1])))
    all_dots[0].grow()
    # print(len(all_dots))
    # show(img=my_map)
    while len(all_dots) > 0:
        for _ in all_dots:
            _.grow()
        # print(len(all_dots))
        # print(occupancy)
        # show(img=my_map)

    cv.imwrite('res/map-grid.jpg', my_map)


def deform(map_=None):
    if map_ is None:
        map_ = 'res/map-grid'
    x = np.linspace(0, 2 * np.pi, MAP_WIDTH + 1)
    shifts = []
    steps = 10
    for i in range(steps):
        y_diff = -1
        while not y_diff > 0:
            y = np.convolve(randint(0, 100) * np.sin(randint(0, 10) * x), randint(0, 100) * np.cos(randint(0, 10) * x),
                            'same')
            # y = randint(1, 2) * np.sin(randint(1, 2) * x) + \
            #    randint(1, 2) * np.sin(randint(1, 2) * x + 2 * np.pi * np.random.rand())
            # print(y.min(), y.max())
            y_diff = y.max() - y.min()
        y_new = 2 * y / y_diff
        shifts.append(y_new)
    print("Shifts are generated")

    old_map = 255 - cv.imread(map_ + '.jpg', cv.IMREAD_GRAYSCALE)
    rows, cols = old_map.shape
    # rot_mat = cv.getRotationMatrix2D((cols / 2, rows / 2), 360 // steps, 1)

    for k in tqdm(range(steps)):
        mod_map = 255 * np.ones((MAP_WIDTH + 1, MAP_WIDTH + 1))
        t_shift = randint(SHIFT_GAP // 2, SHIFT_GAP + 1)
        for i in range(MAP_WIDTH + 1):
            this_shift = shifts[k][i] * t_shift  # randint(SHIFT_GAP//2, SHIFT_GAP + 1)
            for j in range(MAP_WIDTH + 1):
                if 0 <= j + int(this_shift) <= MAP_WIDTH:
                    if k % 2 == 0:
                        mod_map[i, j + int(this_shift)] = old_map[i, j]
                    else:
                        mod_map[j + int(this_shift), i] = old_map[j, i]
        old_map = mod_map.copy()
        rot_mat = cv.getRotationMatrix2D((cols / 2, rows / 2), randint(0, 360), 1)
        # rot_mat = cv.getRotationMatrix2D((cols / 2, rows / 2), 360 // steps, 1)
        old_map = cv.warpAffine(old_map, rot_mat, (cols, rows))
        show(img=mod_map)
    """    
    x_new = [*range(0, MAP_WIDTH + 1)]
    for x_ in x_new:
        mod_map[x_, y_new[x_]] = 0
    """
    mod_map = sharpening(mod_map)
    cv.imwrite(map_ + '-distorted.jpg', 255 - mod_map)


def sharpening(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv.filter2D(src=img.copy(), ddepth=-1, kernel=kernel)
    img = cv.GaussianBlur(img, (15, 15), 0)
    img[img <= 128] = 0
    img[img > 128] = 255
    return img


if __name__ == '__main__':
    generate()
    # img_ = 'test/image001'
    deform()
