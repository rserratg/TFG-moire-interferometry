import numpy as np
from multiprocessing import Pool, cpu_count

def losetime(x):
    print(f"Start: {x}")
    i = 0
    imax = 50e7
    while i < imax:
        i += 1
    print(f"Finish: {x}")
    return x

if __name__ == "__main__":
    print(f"Cpu count: {cpu_count()}")
    with Pool(15) as pool:
        res = pool.map(losetime, range(cpu_count()))
    print(res)
