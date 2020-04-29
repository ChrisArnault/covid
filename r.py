
import random
import numpy as np
import pandas as pd

x = lambda : np.random.random()
y = lambda : np.random.random()

num_vertices = 1000000
num_edges = num_vertices
dist_max = 0.1

class Vertex(object):
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def dist(self, other):
        # print("dist> ", self, other)
        d = np.sqrt(pow(self.x - other.x, 2.0) + pow(self.y - other.y, 2.0))
        # print(d)
        return d




vertices = [ [Vertex(v_id, x(), y()) for v_id in range(num_vertices)], ["id", "x", "y"] ]

def eit(n, last):
    v = 0
    while v < n:
        m = random.randint(0, int(10))
        j = 0
        while j < m:
            w = random.randint(0, n-1)
            # print(v, w)
            if v > last:
                break
            vs = vertices[0]
            # print(len(vs), v, w)
            src = vs[v]
            dest = vs[w]
            j += 1
            if src.dist(dest) < dist_max:
                yield v, w
        if v > last:
            break
        v += 1

edges = [(v, w) for v, w in eit(num_vertices, num_edges)]

print(edges)
