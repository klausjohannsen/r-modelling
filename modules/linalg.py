# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
from copy import deepcopy

##########################################
# support functions
##########################################
def print_formatted_matrix(s, A):
    def mat2str(A):
        return('  ' + str(A).replace('\n','\n  '))
    A = A.astype(float)
    n_mask_true = 0
    if type(A) == np.ma.core.MaskedArray:
        n_mask_true = np.sum(A.mask)
        A = A.filled(np.nan)
    N = 5
    THRESHOLD = 1e-10
    A[np.abs(A) < THRESHOLD] = 0
    SUPPRESS = False
    if A.shape[0] > N or A.shape[1] > N:
        with np.printoptions(precision = 4, suppress = SUPPRESS, threshold= 6, edgeitems = 2):
            print(f'{s} = \n{mat2str(A)}')
            print(f'  ({A.shape[0]} x {A.shape[1]})')
    else:
        with np.printoptions(precision = 4, suppress = SUPPRESS, edgeitems = 2):
            print(f'{s} = \n{mat2str(A)}')
    if n_mask_true:
        print(f'  # masks: {n_mask_true}, {100 * n_mask_true / A.shape[0] / A.shape[0]}%')
    print()

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

def fscp(x, y):
    if type(x) == np.ma.core.MaskedArray:
        s = np.sum(x.filled(0) * y.filled(0))
    else:
        s = np.sum(x * y)
    return(s)

def fnorm(x):
    if type(x) == np.ma.core.MaskedArray:
        return(la.norm(x.filled(0)))
    else:
        return(la.norm(x))

##########################################
# new svd part
##########################################
def iter_coll_1(A, X, Y, s):
    X = X * np.sqrt(s).reshape(1, -1)
    Y = Y * np.sqrt(s).reshape(1, -1)

    M = A.mask
    Dx = (1 - M.astype(int)) @ (X * X)
    Dy = (1 - M.astype(int)) @ (Y * Y)
    XX = ma.dot(A, Y) / Dy
    YY = ma.dot(A.T, X) / Dx

    # update
    X = 0.5 * X + 0.5 * XX.filled(np.nan)
    Y = 0.5 * Y + 0.5 * YY.filled(np.nan)

    # orthogonalize
    for k in range(1, X.shape[1]):
        for kk in range(0, k):
            X[:, k] -= (X[:, k] @ X[:, kk]) / (X[:, kk] @ X[:, kk]) * X[:, kk]
    for k in range(1, Y.shape[1]):
        for kk in range(0, k):
            Y[:, k] -= (Y[:, k] @ Y[:, kk]) / (Y[:, kk] @ Y[:, kk])* Y[:, kk]

    # normalize XX, YY
    X_norm = la.norm(X, axis = 0)
    X /= X_norm.reshape(1, -1)
    Y_norm = la.norm(Y, axis = 0)
    Y /= Y_norm.reshape(1, -1)

    # scale
    s = X_norm * Y_norm

    return(X, Y, s)

def iter_coll_X(A, X, Y, s, theta):
    X = X * np.sqrt(s).reshape(1, -1)
    Y = Y * np.sqrt(s).reshape(1, -1)

    M = A.mask
    Dy = (1 - M.astype(int)) @ (Y * Y)
    XX = ma.dot(A, Y) / Dy

    X = (1 - theta) * X + theta * XX.filled(np.nan)

    # orthogonalize
    for k in range(1, X.shape[1]):
        for kk in range(0, k):
            X[:, k] -= (X[:, k] @ X[:, kk]) / (X[:, kk] @ X[:, kk]) * X[:, kk]

    # normalize XX, YY
    X_norm = la.norm(X, axis = 0)
    X /= X_norm.reshape(1, -1)
    Y_norm = la.norm(Y, axis = 0)
    Y /= Y_norm.reshape(1, -1)

    # scale
    s = X_norm * Y_norm

    return(X, Y, s)

def iter_coll_Y(A, X, Y, s, theta):
    X = X * np.sqrt(s).reshape(1, -1)
    Y = Y * np.sqrt(s).reshape(1, -1)

    M = A.mask
    Dx = (1 - M.astype(int)) @ (X * X)
    YY = ma.dot(A.T, X) / Dx

    Y = theta * Y + theta * YY.filled(np.nan)

    # orthogonalize
    for k in range(1, X.shape[1]):
        for kk in range(0, k):
            Y[:, k] -= (Y[:, k] @ Y[:, kk]) / (Y[:, kk] @ Y[:, kk]) * Y[:, kk]

    # normalize XX, YY
    X_norm = la.norm(X, axis = 0)
    X /= X_norm.reshape(1, -1)
    Y_norm = la.norm(Y, axis = 0)
    Y /= Y_norm.reshape(1, -1)

    # scale
    s = X_norm * Y_norm

    return(X, Y, s)

def iter_coll_2(A, X, Y, s, theta):
    X, Y, s = iter_coll_X(A, X, Y, s, theta)
    X, Y, s = iter_coll_Y(A, X, Y, s, theta)
    return(X, Y, s)

def msvd_coll(A, n = None, max_iter = 1000, tol = 1e-8, verbose = False):
    # copy, as AA will be modified
    AA = deepcopy(A)

    # print iff
    if verbose:
        fn0 = fnorm(AA)
        print('## msvd')
        print(f'0: {fn0:e}')

    # get n-dimensional approximation
    X_0 = 1 - 2 * np.random.rand(A.shape[0], n)
    Y_0 = 1 - 2 * np.random.rand(A.shape[0], n)
    s_0 = np.ones(n)
    AAA_0 = X_0 @ np.diag(s_0) @ Y_0.T
    for k in range(max_iter):

        # iterate
        X_1, Y_1, s_1 = iter_coll_2(AA, X_0, Y_0, s_0, 0.25)
        AAA_1 = X_1 @ np.diag(s_1) @ Y_1.T

        # errors
        AAA_err = fnorm(AAA_1 - AAA_0)
        AAA_err_rel = AAA_err / fn0

        # convergence rate
        if k > 1:
            cr = AAA_err / AAA_err_old
            #print(f'{k} {cr:e}')
        AAA_err_old = AAA_err

        # print
        print(f'iter {k}: err = {AAA_err}, rel err = {AAA_err_rel}')
        if AAA_err_rel < tol:
            break

        # update
        X_0 = X_1
        Y_0 = Y_1
        s_0 = s_1
        AAA_0 = AAA_1

    U = X_1
    VT = Y_1.T
    s = s_1

    # output
    if verbose:
        AA = A - U @ np.diag(s) @ VT
        print(f'{k + 1}: {fnorm(AA):e} {fnorm(AA) / fn0:e}')
        print()

    # return
    return(U, s, VT)

##########################################
# svd
##########################################
def iter_1(A, x, y):
    M = A.mask
    dx = (1 - M.astype(int)) @ (x * x)
    dy = (1 - M.astype(int)) @ (y * y)
    xx = ma.dot(A, y) / dy
    yy = ma.dot(A.T, x) / dx
    xx = 0.5 * (xx.filled(np.nan) + x)
    yy = 0.5 * (yy.filled(np.nan) + y)
    alpha = np.sqrt(la.norm(y) / la.norm(x))
    xx *= alpha
    yy /= alpha
    return(xx, yy)

def iter_2(A, x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    Axy = (A - x @ y.T).filled(0)
    gx = -2 * Axy @ y
    gy = -2 * Axy.T @ x

    # optimal scaling
    for NN in range(5, 100):
        xx = np.linspace(-1, 1, num = NN)
        yy = np.zeros(NN)
        for k in range(NN):
            Axy = A - (x + xx[k] * gx) @ (y + xx[k] * gy).T
            yy[k] = fscp(Axy, Axy)
        c = np.polyfit(xx, yy, 4)
        if c[0] > 0:
            break
    assert(NN < 99)

    roots = np.roots(np.array([4 * c[0], 3 * c[1], 2 * c[2], c[3]]))
    roots = roots[roots.imag == 0].real
    if roots.shape[0] == 1:
        alpha = roots[0]
    elif roots.shape[0] == 3:
        root_1 = np.min(roots)
        root_2 = np.max(roots)
        Axy = A - (x + root_1 * gx) @ (y + root_1 * gy).T
        value_1 = fscp(Axy, Axy)
        Axy = A - (x + root_2 * gx) @ (y + root_2 * gy).T
        value_2 = fscp(Axy, Axy)
        alpha = root_1 if value_1 < value_2 else root_2
    else:
        assert(0)

    # update x, y
    xx = (x + alpha * gx)
    yy = (y + alpha * gy)
    beta = np.sqrt(la.norm(yy) / la.norm(xx))
    xx *= beta
    yy /= beta

    return(xx.reshape(-1), yy.reshape(-1))

def approximate_vector(A, max_iter = 10000, tol = 1e-7):
    n = A.shape[0]
    x = np.random.rand(n)
    y = np.random.rand(n)
    for k in range(max_iter):
        x_new, y_new = iter_1(A, x, y)
        dx = la.norm(x - x_new)
        dy = la.norm(y - y_new)
        dxy_1 = la.norm(x_new - y_new)
        dxy_2 = la.norm(x_new + y_new)
        dxy = min(dxy_1, dxy_2)
        sgn = '+' if dxy_1 < dxy_2 else '-'
        if 0:
            print(f'vector {n}, iteration {k:4d}: dx = {dx}, dy = {dy}, |x-y| = {dxy} ({sgn})')
        if (dx < tol and dy < tol) or k == max_iter - 1:
            break
        x = x_new
        y = y_new
    info = f'[ n_iter = {k:4d}, dx = {dx:e}, dy = {dy:e}, |x-y| = {dxy:e}, ({sgn}) ]'

    sx = la.norm(x)
    sy = la.norm(y)

    return(x / sx, y / sy, sx * sy, info)

def msvd(A, n = None, verbose = False):
    # copy, as AA will be modified
    AA = deepcopy(A)

    # print iff
    if verbose:
        fn0 = fnorm(AA)
        print('## msvd')
        print(f'0: {fn0:e}')

    # get n-dimensional approximation
    U = np.zeros((A.shape[0], 0))
    VT = np.zeros((0, A.shape[0]))
    S = np.zeros((0,0))
    for k in range(n):
        x, y, s, info = approximate_vector(AA)
        U = np.hstack((U, x.reshape(-1, 1)))
        VT = np.vstack((VT, y.reshape(1, -1)))
        S = np.diag(np.hstack((np.diag(S), np.array([s]))))

        # orthogonalize
        if k > 0:
            U = U @ np.sqrt(S)
            VT = np.sqrt(S) @ VT
            W = U.T @ U
            val, Q = la.eig(W)
            U = U @ Q
            VT = Q.T @ VT
            S = np.diag( [ U[:, kk] @ U[:, kk] for kk in range(k + 1) ] )
            inv_sqrt_S = np.diag( [ 1 / np.sqrt(U[:, kk] @ U[:, kk]) for kk in range(k + 1) ] ) 
            U = U @ inv_sqrt_S
            VT = inv_sqrt_S @ VT

        # rest matrix to approximate
        AA = A - U @ S @ VT

        # output
        if verbose:
            print(f'{k + 1}: {fnorm(AA):e} {fnorm(AA) / fn0:e}     {info}')
    if verbose:
        print()

    # return
    return(U, np.diag(S), VT)

def svd(A, n = None, verbose = False):
    if type(A) == np.ndarray:
        # regular svd
        U, s, VT = la.svd(A)
        S = np.diag(s)
        if n is not None and n < A.shape[0]:
            U = U[:, :n]
            S = S[:n, :n]
            VT = VT[:n, :]
            s = s[:n]

    #elif type(A) == np.ma.core.MaskedArray and ma.is_masked(A) == False:
    #    # masked array is not masked, apply regular svd
    #    return(svd(A.filled(np.nan), n = n, verbose = verbose))

    elif type(A) == np.ma.core.MaskedArray:
        # masked svd
        U, s, VT = msvd_coll(A, n = n, verbose = verbose)
        S = np.diag(s)

    else:
        # error: unknown type
        print(type(A))
        assert(0)

    # print
    if verbose:
        A_approx = U @ S @ VT
        print_formatted_matrix('A', A)
        print_formatted_matrix('A_approx', A_approx)
        print_formatted_vector('s', s)
        print(f'rel error = {fnorm(A - A_approx) / fnorm(A)}\n')
        print_formatted_matrix('U.T @ U', U.T @ U)
        print_formatted_matrix('V.T @ V', VT @ VT.T)
        print_formatted_matrix('U.T @ V', U.T @ VT.T)

    # return
    return(U, S, VT)


