# libs
import numpy as np
import matplotlib.pyplot as plt

# functions
def plot2d(X, using = [0, 1]):
    assert(X.shape[1] >= 2)
    assert(len(using) == 2)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(X[:, using[0]], X[:, using[1]])
    plt.axis('equal')
    plt.show()

def plot3d(X, using = [0, 1, 2]):
    assert(X.shape[1] >= 3)
    assert(len(using) == 3)
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    x0 = X[:, using[0]] 
    x1 = X[:, using[1]] 
    x2 = X[:, using[2]] 
    ax.scatter3D(x0, x1, x2)
    l0 = np.max(x0) - np.min(x0)
    l1 = np.max(x1) - np.min(x1)
    l2 = np.max(x2) - np.min(x2)
    dl = max(l0, l1, l2) / 2
    m0 = 0.5 * (np.max(x0) + np.min(x0))
    m1 = 0.5 * (np.max(x1) + np.min(x1))
    m2 = 0.5 * (np.max(x2) + np.min(x2))
    plt.xlim(m0 - dl, m0 + dl)
    plt.ylim(m1 - dl, m1 + dl)
    ax.set_zlim(m2 - dl, m2 + dl)
    plt.show()


