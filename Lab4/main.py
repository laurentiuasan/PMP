import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import arviz as az

"""
client = Poisson(20/60min = 1/3min)
plata = Normal(mu=1/min, sigma=0.5/min)
gatit = Exponential(mu=alfa)
"""

if __name__ == '__main__':
    model = pm.Model()
    alpha = 1

    with model:
        number_clients = pm.Poisson('clients', 20)
        time_for_order = pm.TruncatedNormal('order', mu=1, sigma=0.5, lower=0)
        time_for_serving = pm.Exponential('serving', alpha)
        total_waiting_time = pm.TruncatedNormal('total', mu=time_for_order, sigma=time_for_serving, lower=0)
        trace = pm.sample(10000)

    az.plot_posterior(trace)
    plt.show()
