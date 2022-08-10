# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import modules.relation as rel
import modules.linalg as linalg
from modules.points import circle, xline
from modules.plot import plot2d, plot3d

# run
#X = circle()
#X = xline(shift = [1, 1])
#R = rel.distance_relation(X)
#R = rel.scp_relation(X) 
#R = rel.binary_relation(R, threshold = 0.5, r = '>')

R = rel.countries_1()

U, S, VT = linalg.svd(R, n = 4, verbose = True)
XX = U @ S

plot3d(XX)
