# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import modules.relation_matrix as rm
import copy

# config
N = 100
V = 8

# run
X = np.loadtxt(f'data/circle_{N}.txt')
#R = rm.distance_relation(X, treshold = 4)
R = rm.scp_relation(X, treshold = 0)
#R = rm.neigbor_relation(X, treshold = 0.5)

X, Y, S = R.approximate(V, verbose = 1)
X = np.diag(S) @ X

import matplotlib.pyplot as plt

if V > 2:
    # 3D
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X[0], X[1], X[2])
    l0 = np.max(X[0]) - np.min(X[0])
    l1 = np.max(X[1]) - np.min(X[1])
    l2 = np.max(X[2]) - np.min(X[2])
    dl = max(l0, l1, l2) / 2
    m0 = 0.5 * (np.max(X[0]) + np.min(X[0]))
    m1 = 0.5 * (np.max(X[1]) + np.min(X[1]))
    m2 = 0.5 * (np.max(X[2]) + np.min(X[2]))
    plt.xlim(m0 - dl, m0 + dl)
    plt.ylim(m1 - dl, m1 + dl)
    ax.set_zlim(m2 - dl, m2 + dl)
else:
    # 2d
    plt.scatter(X[0], X[1])

plt.show()


