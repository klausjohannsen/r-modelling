# libs
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# fcts
a = -1
theta = 0.5
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
    conv_rate = np.zeros((10000, 2))
    for k in range(10000):
        xx = f2(x)
        e = la.norm(x - xx)
        if k > 0:
            conv_rate[k, 0] = k
            conv_rate[k, 1] = e / e_old
        e_old = e
        x = xx
        if e == 0:
            return(conv_rate[1:(k + 1)])

# run

plt.figure(figsize=(20, 10), dpi=80)
plt.yscale('log')
for theta in list(np.linspace(0.32, 0.35, num = 4)):
    x = 1 - 2 * np.random.rand(2)
    cr = test(x)
    print(theta, cr.shape)
    plt.plot(cr[1:, 0], cr[1:, 1], label = f'theta = {theta}')
plt.legend()
plt.show()


