# libs
import numpy as np
import numpy.linalg as la

# config
N = 1000

# run
x = 1 - 2 * np.random.rand(N, 3)
x /= la.norm(x, axis = 1).reshape(-1, 1)

# save
np.savetxt(f'data/sphere_{N}.txt', x)


