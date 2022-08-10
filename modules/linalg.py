# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma

##########################################
# support functions
##########################################
def print_formatted_matrix(s, A):
    def mat2str(A):
        return('  ' + str(A).replace('\n','\n  '))
    A = A.astype(float)
    N = 5
    THRESHOLD = 1e-10
    A[np.abs(A) < THRESHOLD] = 0
    SUPPRESS = False
    if A.shape[0] > N or A.shape[1] > N:
        with np.printoptions(precision = 4, suppress = SUPPRESS, threshold= 6, edgeitems = 2):
            print(f'{s} = \n{mat2str(A)}')
            print(f'  ({A.shape[0]} x {A.shape[1]})\n')
    else:
        with np.printoptions(precision = 4, suppress = SUPPRESS, edgeitems = 2):
            print(f'{s} = \n{mat2str(A)}\n')

def print_formatted_vector(s, v):
    v = v.astype(float)
    N = 5
    THRESHOLD = 1e-10
    v[np.abs(v) < THRESHOLD] = 0
    if v.shape[0] > N:
        ss = [f'{v[0]:.4e}, {v[1]:.4e}, {v[2]:.4e}, {v[3]:.4e}, ...']
        print(f'{s} = [{", ".join(ss)}]\n')
    else:
        ss = [f'{v[k]:.4e}' for k in range(v.shape[0])]
        print(f'{s} = [{", ".join(ss)}]\n')

##########################################
# svd
##########################################
def svd(A, n = None, verbose = False):
    U, s, VT = la.svd(A)
    S = np.diag(s)
    if n is not None and n < A.shape[0]:
        U = U[:, :n]
        S = S[:n, :n]
        VT = VT[:n, :]
        s = s[:n]

    # print
    if verbose:
        A_approx = U @ S @ VT
        print_formatted_matrix('A', A)
        print_formatted_matrix('A_approx', A_approx)
        print_formatted_vector('s', s)
        print(f'rel error = {la.norm(A - A_approx) / la.norm(A)}\n')
        print_formatted_matrix('U.T @ U', U.T @ U)
        print_formatted_matrix('V.T @ V', VT @ VT.T)
        print_formatted_matrix('U.T @ V', U.T @ VT.T)

    # return
    return(U, S, VT)


