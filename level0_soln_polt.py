import numpy as np
import matplotlib.pyplot as plt

data_f = open("level0_curve.txt", "r")
nx0 = 25
xs = []; h0 = []
for line in data_f:
	x, h = list(map(float, line.split()))
	xs.append(x)
	h0.append(h)
xs = np.array(xs)
h0 = np.array(h0)

plt.plot(xs, h0)
plt.show()