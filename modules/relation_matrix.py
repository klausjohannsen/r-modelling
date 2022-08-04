# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import copy
from scipy.spatial import distance_matrix 
from scipy.linalg import svdvals

# functions
def dist_matrix(x, p = 2):
    D = distance_matrix(x, x, p = p)
    return(D)

def dist_mask(D, treshold = 1, mask_larger = True):
    if mask_larger:
        return(D > treshold)
    else:
        return(D < treshold)

def frobenius_scp(x, y):
    s = np.sum(x.filled(0) * y.filled(0))
    return(s)

# base class relation
class relation:
    def __init__(self, t = None, create_substructures = True):
        self.t = t
        self.xl = []
        self.yl = []
        self.sl = []
        self.A_partial_list = []
        self.A_masked_partial_list = []

        if create_substructures:
            # transpose
            self.T = relation(t + ', transposed', create_substructures = False)
            self.T.R_full = self.R_full.T
            self.T.M = self.M.T
            self.T.R = self.R.T
            self.T.n = self.n
            self.T.T = self

    def __str__(self):
        if self.t is None:
            return('')
        s = f'# {self.t}\n'
        s += f'n = {self.n}\n\n'
        s += f'R_full =\n{self.R_full}\n\n'
        s += f'M =\n{self.M}\n\n'
        s += f'R =\n{self.R.filled(np.nan)}\n\n'
        s += f'SV    = {svdvals(self.R_full)[:4]}'
        if len(self.sl):
            s += f'\nSV(a) = {np.array(self.sl)}'
            A = copy.deepcopy(self.R)
            fn = [la.norm(A.filled(0)) ** 2]
            n = len(self.sl)
            for k in range(n):
                A -= self.sl[k] * self.xl[k].reshape(-1, 1) @ self.yl[k].reshape(1, -1)
                fn += [la.norm(A.filled(0)) ** 2]
            s += f'\nFN = {np.array(fn)}\n\n'
            corr = np.zeros((n, n))
            frobenius_norm = np.zeros(n)
            for k in range(n):
                frobenius_norm[k] = np.sqrt(frobenius_scp(self.A_masked_partial_list[k], self.A_masked_partial_list[k]))
            for k in range(n):
                for kk in range(n):
                    corr[k, kk] = frobenius_scp(self.A_masked_partial_list[k], self.A_masked_partial_list[kk]) / frobenius_norm[k] / frobenius_norm[kk]
            s += f'corr = \n{corr}\n\n'
        return(s)

    def D(self, x):
        d = (1 - self.M.astype(int)) @ (x * x)
        return(d)

    def iter(self, A, x, y):
        dx = self.D(x)
        dy = self.D(y)
        xx = ma.dot(A, y) / dy
        yy = ma.dot(A.T, x) / dx
        xx = 0.5 * (xx.filled(np.nan) + x)
        yy = 0.5 * (yy.filled(np.nan) + y)
        alpha = np.sqrt(la.norm(y) / la.norm(x))
        xx *= alpha
        yy /= alpha
        return(xx, yy)

    def approximate_vector(self, A, n, max_iter = 10000, tol = 1e-9, verbose = 1):
        x = np.random.rand(self.n)
        y = np.random.rand(self.n)
        for k in range(max_iter):
            x_new, y_new = self.iter(A, x, y)
            dx = la.norm(x - x_new)
            dy = la.norm(y - y_new)
            dxy = la.norm(x_new - y_new)
            if verbose == 2:
                print(f'vector {n}, iteration {k}: dx = {dx}, dy = {dy}, |x-y| = {dxy}')
            if (dx < tol and dy < tol) or k == max_iter - 1:
                break
            x = x_new
            y = y_new
        if verbose:
            print(f'vector {n}: dx = {dx}, dy = {dy}, |x-y| = {dxy}')
        sx = la.norm(x)
        sy = la.norm(y)
        return(x / sx, y / sy, sx * sy)

    def approximate(self, n, verbose = 1):
        A = copy.deepcopy(self.R)
        for k in range(n):
            x, y, s = self.approximate_vector(A, k, verbose = verbose)
            self.xl += [x]
            self.yl += [y]
            self.sl += [s]
            A_partial = s * x.reshape(-1, 1) @ y.reshape(1, -1)
            self.A_partial_list += [A_partial]
            self.A_masked_partial_list += [ma.array(A_partial, mask = self.M)]
            A -= self.A_partial_list[-1]

        X = np.vstack(self.xl)
        Y = np.vstack(self.yl)
        S = np.array(self.sl)

        return(X, Y, S)

# derived classes
class distance_relation(relation):
    def __init__(self, x, p = 2, treshold = 1, mask_larger = True):
        self.R_full = dist_matrix(x, p = p)
        self.M = dist_mask(self.R_full, treshold = treshold, mask_larger = mask_larger)
        self.R = ma.array(self.R_full, mask = self.M)
        self.n = self.R.shape[0]

        # base call initialization
        super(distance_relation, self).__init__(t = 'distance_relation')

class scp_relation(relation):
    def __init__(self, x, p = 2, treshold = 0):
        self.R_full = x @ x.T
        self.M = dist_mask(self.R_full, treshold = treshold, mask_larger = False)
        self.R = ma.array(self.R_full, mask = self.M)
        self.n = self.R.shape[0]

        # base call initialization
        super(scp_relation, self).__init__(t = 'scp_relation')

class neigbor_relation(relation):
    def __init__(self, x, p = 2, treshold = 0):
        self.R_full = dist_matrix(x, p = p)
        self.M = self.R_full > 2 * treshold
        self.R_full = self.R_full < treshold
        self.R_full = self.R_full.astype(float)
        self.R = ma.array(self.R_full, mask = self.M)
        self.n = self.R.shape[0]

        # base call initialization
        super(neigbor_relation, self).__init__(t = 'neigbor_relation')
