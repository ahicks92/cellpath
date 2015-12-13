from __future__ import division, print_function
import numpy as np
import numba
from numba import cuda
import random
import sys
import time
import math
import pdb

width = 32768
height = 16384

def make_grid():
    arr = np.random.rand(width, height)
    arr = arr*0.9+0.1
    return arr.astype(np.float32)

def pick_start():
    return width//2, height//2
inf = float('infinity')

@cuda.jit(device=True)
def cell(grid, costs, x, y, dx, dy, epsilon):
    v1 = grid[x][y]
    c1 = inf
    c2 = inf
    c3 = inf
    c4 = inf
    if x-1 >= 0:
        c1 = grid[x-1][y]+costs[x-1][y]
    if x+1 < dx:
        c2 = grid[x+1][y]+costs[x+1][y]
    if y-1 >= 0:
        c3 = grid[x][y-1]+costs[x][y-1]
    if y+1 < dy:
        c4 = grid[x][y+1]+costs[x][y+1]
    v2 = min(v1, min(min(c1, c2), min(c3, c4)))
    writing = v1-v2 >= epsilon
    grid[x][y] = v2
    return cuda.selp(writing, 1.0, 0.0)

@cuda.jit(fastmath=True)
def kernel(grid, costs, written, dx, dy, kx, ky, epsilon, limiter):
    running = True
    writes = numba.float32(0)
    x, y = cuda.grid(2)
    while running and limiter:
        limiter -= 1
        wrote_this_time = numba.float32(0)
        for j in range(ky):
            for i in range(kx):
                wrote_this_time += cell(grid, costs, x*kx+i, y*ky+j, dx, dy, epsilon)
        writes += wrote_this_time
        runnin = wrote_this_time > 0
    cuda.atomic.add(written, (cuda.blockIdx.x, cuda.blockIdx.y), writes)

@numba.jit
def initial_pass(grid, costs, dx, dy):
    for y in range(dy):
        for x in range(1, dx):
            prev = grid[x-1][y]
            cur = grid[x][y]
            grid[x][y] = min(cur, prev+costs[x-1][y])
    for x in range(dx):
        for y in range(1, dy):
            prev = grid[x][y-1]
            cur = grid[x][y]
            grid[x][y] = min(cur, prev+costs[x][y-1])

#A workspace. Every path is infinitely long to start.
grid = np.full((width, height), float('infinity'), dtype = np.float32)
costs = make_grid()
epsilon = np.min(costs)/2.0
print(epsilon)
start = pick_start()
grid[start[0]][start[1]] = 0.0 #Going to the start is free.

stream = cuda.stream()
dgrid = cuda.to_device(grid, stream)
dcosts = cuda.to_device(costs, stream)

threadX = 16
threadY = 16
blockX = 512
blockY = 512
widthPerKernel = width//threadX//blockX
heightPerKernel = height//threadY//blockY

initial_pass(grid, costs, width, height)
written = np.zeros((blockX, blockY), dtype=np.float32)
print("Running. Start =", start)
begin = time.time()
first = True
writes = 0
writes_next = 0
limits = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 21, 23]
lindex = 0
while writes < writes_next or first:
    first = False
    writes = writes_next
    for lim in limits:
        kernel[(blockX, blockY), (threadX, threadY), stream](dgrid, dcosts, written, width, height, widthPerKernel, heightPerKernel, epsilon, lim)
        lindex += 1
    stream.synchronize()
    writes_next = int(written.sum())
    print(writes_next-writes)
end = time.time()
print("Total time:", end-begin)
print("Wrote", writes)