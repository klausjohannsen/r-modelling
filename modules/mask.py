# libs
import numpy as np

# functions
def mask_distance(R, threshold = 0, r = None):
    assert(r == '>' or r == '<')
    if r == '<':
        R = R < threshold
    if r == '>':
        R = R > threshold
    return(R)

def mark_random(R, mode = None, symmetric = True, diagonals = False):
    n = R.shape[0]
    M = np.zeros((n,n), dtype = bool)
    if mode is None:
        return(M)
    if mode[0] == "#":
        nn = mode[1]
    if mode[0] == "%":
        nn = int(0.01 * mode[1] * n * n)

    nnn = 0
    while(nnn < nn):
        i, j = np.random.randint(n, size = 2)
        if M[i, j] == True: continue
        if i == j and diagonals == False: continue
        if symmetric == False and diagonals == False:
            M [i, j] = True
            nnn += 1
        if symmetric == True and diagonals == False:
            M [i, j] = True
            M [j, i] = True
            nnn += 2
        if symmetric == False and diagonals == True:
            M [i, j] = True
            nnn += 1
        if symmetric == True and diagonals == True:
            M [i, j] = True
            nnn += 1
            if i != j:
                M [j, i] = True
                nnn += 1
    return(M)





