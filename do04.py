# libs
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import modules.relation_matrix as rm
import copy

# config
N = 100

# run
X = np.loadtxt(f'data/circle_{N}.txt')
R = rm.distance_relation(X, treshold = 10)

print("#########################################")
print(R)
print("#########################################")

# iteration init
x = np.random.rand(N)
#y = np.random.rand(N)
y = copy.deepcopy(x)

# first vector
NN = 300
for k in range(NN):
    x_new, y_new = R.iter(x, y)
    if k == NN - 1:
        print('first vector, last iteration:', la.norm(x - x_new), la.norm(y - y_new), la.norm(x_new - y_new))
    x = x_new
    y = y_new
x0 = x
y0 = y

X0 = R.R
X0n = ma.sum(X0 * X0)
print("## X0", X0n)
X1 = x0.reshape(-1,1) @ y0.reshape(1,-1)
X1 = ma.array(X1, mask = R.M)
X1n = ma.sum(X1 * X1)
print("## X1", X1n, X1n / X0n)
X2 = X1 - X0
X2n = ma.sum(X2 * X2)
print("## X2", X2n, X2n / X0n)
print()

# second vector
def otho(x, y):
    y = y / la.norm(y)
    return(x - np.dot(x, y) * y)

x = otho(np.random.rand(N), x0)
#y = otho(np.random.rand(N), y0)
y = copy.deepcopy(x)
NN = 1000
for k in range(NN):
    x_new, y_new = R.iter(x, y)
    x_new = otho(x_new, x0)
    y_new = otho(y_new, y0)
    if k == NN - 1:
        print('second vector, last iteration:', la.norm(x - x_new), la.norm(y - y_new), la.norm(x_new - y_new))
    x = x_new
    y = y_new
x1 = x
y1 = y

print("## X0", X0n)
print("## X1", X1n, X1n / X0n)
X2 = x1.reshape(-1,1) @ y1.reshape(1,-1)
X2 = ma.array(X2, mask = R.M)
X2n = ma.sum(X2 * X2)
print("## X2", X2n, X2n / X0n)
X3 = X0 - X1 - X2
X3n = ma.sum(X3 * X3)
print("## X3", X3n, X3n / X0n)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(x0, x1)
plt.show()

