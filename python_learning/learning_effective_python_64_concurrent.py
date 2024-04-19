
import learning_effective_python_64_my_module
import time
from concurrent.futures import ProcessPoolExecutor

NUMBERS = [
    (1963309, 2265973), (2030677, 3814172),
    (1551645, 2229620), (2039045, 2020802),
    (1823712, 1924928), (2293129, 1020491),
    (1281238, 2273782), (3823812, 4237281),
    (3812741, 4729139), (1292391, 2123811)
]


def main():
    start = time.time()
    pool = ProcessPoolExecutor(max_workers=2)
    results = list(pool.map(learning_effective_python_64_my_module.gcd, NUMBERS))
    end = time.time()
    delta = end - start
    print(f'ProcessPoolExecutor took {delta:3f} seconds')
    print(f'Results:{results}')


if __name__ == '__main__':
    main()

