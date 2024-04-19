

import asyncio
import random
import time


async def calculate(num):
    start = time.time()
    delay = random.uniform(0.5, 1)
    await asyncio.sleep(delay*5)
    end = time.time()
    return end - start, num * num


async def go_calc():
    tasks = []
    batch = 10
    batch_index = 0
    batch_no = 0
    batch_sum = 0
    for i in range(200):
        task = calculate(i)
        tasks.append(task)
        batch_index += 1
        if batch_index == batch:
            batch_result = await asyncio.gather(*tasks)
            # batch_sum = sum(batch_result)
            print(f'Batch no {batch_no}, batch_list:{sorted(batch_result)}')
            batch_no += 1
            batch_index = 0
            batch_sum = 0
            tasks = []


asyncio.run(go_calc())
