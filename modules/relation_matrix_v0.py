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

def frobenius_norm(x):
    return(np.sqrt(frobenius_scp(x, x)))

# base class relation
class relation:
    def __init__(self, t = None, create_substructures = True):
        self.t = t
        self.xl = []
        self.yl = []
        self.sl = []

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
        return(s)

    def D(self, x):
        d = (1 - self.M.astype(int)) @ (x * x)
        return(d)

    def iter_1(self, A, x, y):
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

    def iter_2(self, A, x, y):
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
                yy[k] = frobenius_scp(Axy, Axy)
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
            value_1 = frobenius_scp(Axy, Axy)
            Axy = A - (x + root_2 * gx) @ (y + root_2 * gy).T
            value_2 = frobenius_scp(Axy, Axy)
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

    def approximate_vector(self, A, max_iter = 10000, tol = 1e-9):
        x = np.random.rand(self.n)
        y = np.random.rand(self.n)
        for k in range(max_iter):
            x_new, y_new = self.iter_2(A, x, y)
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

    def approximate(self, n, verbose = 1):
        # init
        A = copy.deepcopy(self.R)
        if verbose:
            fn0 = frobenius_norm(A)
            print(f'0: {fn0:e}')
        
        for k in range(n):
            # vector
            x, y, s, info = self.approximate_vector(A)
            self.xl += [x]
            self.yl += [y]
            self.sl += [s]
            A -= s * x.reshape(-1, 1) @ y.reshape(1, -1)

            # output
            if verbose:
                print(f'{k + 1}: {frobenius_norm(A):e} {frobenius_norm(A) / fn0:e}     {info}')

        # return
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
