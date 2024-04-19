"""

第59条 如果必须用线程作并发，那就考虑通过ThreadPoolExecutor来实现

"""


from concurrent.futures import ThreadPoolExecutor
from threading import Lock

ALIVE = '*'
EMPTY = '_'


class Grid:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.rows = []
        self.round = 0
        for _ in range(self.height):
            self.rows.append([EMPTY] * self.width)

    def get(self, y, x):
        return self.rows[y % self.height][x % self.width]

    def set(self, y, x, state):
        self.rows[y % self.height][x % self.width] = state

    def __str__(self):
        result = ''
        for y in range(self.height):
            for x in range(self.width):
                result = result + self.get(y, x)
            result = result + '|'
        return result


def count_neighbors(y, x, get):
    n_ = get(y-1, x+0)
    ne = get(y-1, x+1)
    e_ = get(y+0, x+1)
    se = get(y+1, x+1)
    s_ = get(y+1, x+0)
    sw = get(y+1, x-1)
    w_ = get(y+0, x-1)
    nw = get(y-1, x-1)

    neighbors_states = [n_, ne, e_, se, s_, sw, w_, nw]
    count = 0
    for state in neighbors_states:
        if state == ALIVE:
            count += 1

    return count


def game_logic(state, neighbors):
    if state == ALIVE:
        if neighbors < 2:
            return EMPTY
        elif neighbors >3:
            return EMPTY
    else:
        if neighbors == 3:
            return ALIVE
    return state


def step_cell(y, x, get, set):
    state = get(y, x)
    neighbors = count_neighbors(y, x, get)
    next_state = game_logic(state, neighbors)
    set(y, x, next_state)


class SimulationError(Exception):
    pass


def game_logic_thread(item):
    y, x, state, neighbors = item
    try:
        next_state = game_logic(state, neighbors)
    except Exception as e:
        next_state = e
    return y, x, next_state


class LockingGrid(Grid):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.lock = Lock()

    def __str__(self):
        with self.lock:
            return super().__str__()

    def get(self, y, x):
        with self.lock:
            return super().get(y, x)

    def set(self, y, x, state):
        with self.lock:
            return super().set(y, x, state)


class ExperimentPool:
    def __init__(self, round_num, height, width):
        self.round_num = round_num
        self.height = height
        self.width = width
        self.grids = []

    def go(self, init_grid):
        self.grids.append(init_grid)
        with ThreadPoolExecutor(max_workers=10) as pool:
            for i in range(1, self.round_num):
                next_grid = simulate_pool(pool, self.grids[i-1])
                next_grid.round = i
                self.grids.append(next_grid)

    def render_result(self):
        result = ''
        header = ''
        for r in range(self.round_num):
            ll = len(str(r))
            header += ' ' * (int((self.width - ll)/2)) + str(r) + ' ' * (self.width - ll - int((self.width - ll)/2)) + '|'

        for i in range(self.height):
            for j in range(self.round_num):
                for k in range(self.width):
                    result += self.grids[j].get(i, k)
                result += '|'
            result += '\n'

        result = header + '\n' + result
        return result


def simulate_pool(pool, grid):
    next_grid = LockingGrid(grid.height, grid.width)
    futures = []
    for y in range(grid.height):
        for x in range(grid.width):
            args = (y, x, grid.get, next_grid.set)
            future = pool.submit(step_cell, *args)  # Fan out
            futures.append(future)

    for future in futures:
        future.result()  #Fan in

    return next_grid


init_grid = Grid(5, 9)
init_grid.set(0,3,ALIVE)
init_grid.set(1,4, ALIVE)
init_grid.set(2,2, ALIVE)
init_grid.set(2,3, ALIVE)
init_grid.set(2,4, ALIVE)


experiment = ExperimentPool(15, 5, 9)
experiment.go(init_grid)
print(experiment.render_result())

