# libs
import numpy as np
import matplotlib.pyplot as plt

# config
N = 100

# run
x = np.loadtxt(f'data/circle_{N}.txt')

# viz
plt.figure(figsize = (10, 10), dpi=80)
plt.scatter(x[:, 0], x[:, 1], s = 10)
plt.show()

