import arviz as az

import numpy as np
import pymc3 as pm
import pandas as pd
from theano import tensor as tt

if __name__ == "__main__":
    # 1
    clusters = 3
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 0, 3]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster),
                           np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))

    # 2
    with pm.Model() as model_mgp:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=np.array([.9, 1, .7]) * mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        order_means = pm.Potential('order_means', tt.switch(means[1] - means[0] < 0, -np.inf, 0))
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        idata_mgp = pm.sample(1000, random_seed=123, return_inferencedata=True)

    varnames = ['means', 'p']
    az.plot_trace(idata_mgp, varnames)

    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(mix.min(), mix.max(), cluster),
                              sd=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)

            sd = pm.HalfNormal('sd', sd=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
            idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)
