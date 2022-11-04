import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas
import arviz as az


if __name__ == '__main__':
    data = pandas.read_csv('data.csv')

    with pm.Model() as model:
        x = data.get('momage')
        # calculating mean, sd
        media = x.sum() / x.count()
        deviatia = [(i - media) ** 2 for i in x]
        sd = (sum(deviatia) / len(deviatia)) ** 0.5
        # model
        alpha = pm.Normal('alpha', mu=media, sd=sd)
        beta = pm.Normal('beta', mu=media, sd=sd)
        eps = pm.HalfCauchy('eps', 5)

        y = data.get('ppvt')
        y_pred = pm.Normal('y_pred', mu=media, sd=eps, observed=y)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(x, y)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    plt.show()

