# libs
import numpy as np
from sklearn.manifold import TSNE

# functions
def tsne(X, n = 2, verbose = True):
    tsne_ = TSNE(n_components = 2, verbose = verbose, random_state=123)
    Y = tsne_.fit_transform(X)
    return(Y)





