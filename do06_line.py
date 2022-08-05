# libs
import numpy as np
import numpy.linalg as la

# config
N = 1000

# run
x = 3 - 6 * np.random.rand(N, 2)
x[:,1] = x[:,0]
x += np.array([-1, 1]).reshape(1, 2)

# save
np.savetxt(f'data/line_{N}.txt', x)


