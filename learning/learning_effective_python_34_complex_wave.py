import math


def wave(amplitude_it, steps):
    step_size = 2 * math.pi / steps
    for step in range(steps):
        radian = step * step_size
        amplitude = next(amplitude_it)
        output = amplitude * math.sin(radian)
        yield output


def complex_wave(amplitude_it):
    yield from wave(amplitude_it, 3)
    yield from wave(amplitude_it, 4)
    yield from wave(amplitude_it, 5)


def transmit(output):
    print(f'The output signal:{output}')


def run(func):
    amplitudes = [3, 3, 3, 6, 6, 6, 6, 8, 8, 8, 8, 8]
    # 生成器初始化，先获取生成器，然后使用next或者send驱动生成器
    it = func(iter(amplitudes))
    for _ in amplitudes:
        output = next(it)
        transmit(output)


run(complex_wave)
