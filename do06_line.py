# libs
import numpy as np
import numpy.linalg as la

# config
N = 100

# run
x = 5 - 10 * np.random.rand(N, 2)
x[:,1] = x[:,0]

# save
np.savetxt(f'data/line_{N}.txt', x)


