# libs
import numpy as np
import numpy.linalg as la

# config
N = 1000

# run
x = 1 - 2 * np.random.rand(N, 2)
x /= la.norm(x, axis = 1).reshape(-1, 1)
x += np.array([2, 2]).reshape(1, 2)

# save
np.savetxt(f'data/circle_{N}.txt', x)


