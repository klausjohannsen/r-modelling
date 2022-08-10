# libs
import numpy as np
import numpy.linalg as la

# functions
def circle(n = 100, radius = 1, shift = [0, 0]):
    x = 1 - 2 * np.random.rand(n, 2)
    x /= la.norm(x, axis = 1).reshape(-1, 1)
    x += np.array(shift).reshape(1, 2)
    return(x)

def xline(n = 100, length = 1, shift = [0, 0]):
    x = 0.5 - np.random.rand(n, 2)
    x[:, 1] = 0
    x += np.array(shift).reshape(1, 2)
    return(x)





