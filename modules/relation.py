# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
from scipy.spatial import distance_matrix

# functions
def distance_relation(X, p = 2):
    R = distance_matrix(X, X, p = p)
    return(R)

def scp_relation(X):
    R = np.dot(X, X.T)
    return(R)

def binary_relation(R, threshold = 0, r = None):
    assert(r == '>' or r == '<')
    if r == '<':
        R = 1.0 * (R < threshold)
    if r == '>':
        R = 1.0 * (R > threshold)
    return(R)

def countries_1():
    zambia      = [4,0,0,0,0,2,0,0,0,0]
    turkey      = [0,4,0,0,3,0,1,1,2,1]
    venezuela   = [0,0,4,1,0,0,1,1,0,0]
    usa         = [0,0,1,4,0,1,0,0,0,1]
    saudiarabia = [0,3,0,0,4,0,1,1,2,1]
    southafrica = [2,0,0,1,0,4,0,1,0,2]
    luxembourg  = [0,1,1,0,1,0,4,3,1,2]
    italy       = [0,1,1,0,1,1,3,4,1,3]
    hongkong    = [0,2,0,0,2,0,1,1,4,1]
    denmark     = [0,1,0,1,1,2,2,3,1,4]

    c = [zambia, turkey, venezuela, usa, saudiarabia, southafrica, luxembourg, italy, hongkong, denmark]
    return(np.array(c).astype(float))
