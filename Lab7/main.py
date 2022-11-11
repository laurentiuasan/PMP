import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    # fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    # axes[0, 0].scatter(speed, price, alpha=0.6)
    # axes[0, 1].scatter(hardDrive, price, alpha=0.6)
    # axes[1, 0].scatter(ram, price, alpha=0.6)
    # axes[1, 1].scatter(premium, price, alpha=0.6)
    # axes[0, 0].set_ylabel("Price")
    # axes[0, 0].set_xlabel("Speed")
    # axes[0, 1].set_xlabel("HardDrive")
    # axes[1, 0].set_xlabel("Ram")
    # axes[1, 1].set_xlabel("Premium")
    # plt.savefig('price_correlations.png')

    model = pm.Model()

    with model:
        # weak a priori information
        alpha = pm.Normal('a', mu=0, sd=10)
        beta1 = pm.Normal('b1', mu=0, sd=10)
        beta2 = pm.Normal('b2', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        mu = pm.Deterministic('mu', alpha +
                              beta1 * speed +
                              beta2 * np.log(ram))
        price = pm.Normal('y',
                      mu=mu,
                      sd=sigma,
                      observed=price)

        step = pm.Slice()
        trace = pm.sample(1000, tune=1000, cores=4, step=step)

        alpha_mean = trace['a'].mean().items()
        beta1_mean = trace['b1'].mean().items()
        beta2_mean = trace['b2'].mean().items()

        ppp = pm.sample_posterior_predictive(trace, samples=100, model=model)
        plt.plot((speed, np.log(ram)), alpha_mean + beta1_mean * speed + beta2_mean * np.log(ram))
        sig = az.plot_hdi((speed, np.log(ram)), ppp['y'], hdi_prob=0.97, color='k')
        plt.xlabel("Speed, ln(Ram)")
        plt.ylabel("Score", rotation=0)
        plt.savefig('a.png')




