"""
第34条 不要用send给生成器注入数据
"""

import math


def wave(amplitude, steps):
    step_size = 2 * math.pi / steps
    for step in range(steps):
        radians = step * step_size
        fraction = math.sin(radians)
        output = amplitude * fraction
        yield output


def wave_modulating(steps):
    step_size = 2 * math.pi / steps
    amplitude = yield
    for step in range(steps):
        radians = step * step_size
        fraction = math.sin(radians)
        output = amplitude * fraction
        # 每一个yield都是一个返回入口点，这里赋值的意义在于send把新的值带回来
        amplitude = yield output


def run_modulating(it):
    amplitudes = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for amplitude in amplitudes:
        output = it.send(amplitude)
        transmit(output)


def transmit(output):
    if output is None:
        print(f'Output is None!')
    else:
        print(f'Output:{output:>5.1f}')


def run(it):
    for output in it:
        transmit(output)


# run(wave(3.0, 8))
run_modulating(wave_modulating(12))
