import numpy as np
import numpy.random
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


m1 = np.random.exponential(1/4, size=10000)
m2 = np.random.exponential(1/6, size=10000)

x = 0.4*m1 + 0.6*m2

az.plot_posterior({'m1': m1, 'm2':m2, 'x':x})
plt.show()
