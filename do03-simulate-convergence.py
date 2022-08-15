# libs
import numpy as np
import numpy.linalg as la

# fcts
a = 100
theta = 0.35
def f1(x):
    xx = a * x[1] / x[0] / x[0]
    yy = a * x[0] / x[1] / x[1]
    x = (1 - theta) * x + theta * np.array([xx, yy])
    return(x)

def f2(x):
    x0 = x[0]
    x1 = x[1]

    xx = a * x1 / x0 / x0
    x0 = (1 - theta) * x0 + theta * xx
    yy = a * x0 / x1 / x1
    x1 = (1 - theta) * x1 + theta * yy

    return(np.array([x0, x1]))

def test(x):
    for k in range(10000):
        xx = f2(x)
        e = la.norm(x - xx)
        x = xx
        if e == 0:
            return(1, k)
    return(0, k)

# run
N = 1000
res = np.zeros((0, 2))
cnt = np.zeros(N)
for k in range(N):
    x = 1 - 2 * np.random.rand(2)
    r, cnt[k] = test(x)
    if r == 0:
        print(x)
        res = np.vstack((res, x))

print(f'avg = {np.mean(cnt):e}, std = {np.std(cnt, ddof = 1) / np.sqrt(N):e}')

np.savetxt("res", res)
