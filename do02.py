# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import modules.relation as rel
import modules.linalg as linalg
from modules.points import circle, xline
from modules.plot import plot2d, plot3d
from modules.mask import mask_distance, mark_random

# run
X = circle(n = 1000)
R = rel.distance_relation(X)
M = mark_random(R, mode = ['%', 0])
R = ma.array(R, mask = M)

U, S, VT = linalg.svd(R, n = 4, verbose = True)
XX = U @ S

plot3d(XX)

