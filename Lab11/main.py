import random

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


if __name__ == "__main__":

    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()

    # ex 1.a
    with pm.Model() as model_p:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    # 1.b
    with pm.Model() as model_p_b1:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=100, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred_b1 = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
        idata_p_b1 = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p_b2:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred_b2 = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
        idata_p_b2 = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

    alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C1', label=f'model order {order}')

    # pentru modelul cu sd=100
    alpha_p_post_b1 = idata_p_b1.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post_b1 = idata_p_b1.posterior['beta'].mean(("chain", "draw")).values
    idx_b1 = np.argsort(x_1s[0])
    y_p_post_b1 = alpha_p_post_b1 + np.dot(beta_p_post_b1, x_1s)

    plt.plot(x_1s[0][idx_b1], y_p_post_b1[idx_b1], 'C2', label=f'model order {order}')

    # pentru modelul cu sd=np.array([...])
    alpha_p_post_b2 = idata_p_b1.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post_b2 = idata_p_b1.posterior['beta'].mean(("chain", "draw")).values
    idx_b2 = np.argsort(x_1s[0])
    y_p_post_b2 = alpha_p_post_b2 + np.dot(beta_p_post_b2, x_1s)

    plt.plot(x_1s[0][idx_b2], y_p_post_b2[idx_b2], 'C3', label=f'model order {order}')

    plt.scatter(x_1s, y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    # ex 2
    x_2 = np.random.random_sample(size=500)
    y_2 = np.random.random_sample(size=500)
    order = 5
    x_2p = np.vstack([x_2 ** i for i in range(1, order + 1)])
    x_2s = (x_2p - x_2p.mean(axis=1, keepdims=True)) / x_2p.std(axis=1, keepdims=True)
    y_2s = (y_2 - y_2.mean()) / y_2.std()
    plt.scatter(x_2s[0], y_2s)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()

    # ex 2
    with pm.Model() as model_p2:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alpha + pm.math.dot(beta, x_2s)
        y_pred_2 = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_2s)
        idata_p_2 = pm.sample(2000, return_inferencedata=True)

    x_new_2 = np.linspace(x_2s[0].min(), x_2s[0].max(), 100)

    alpha_p_post_2 = idata_p.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post_2 = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx_2 = np.argsort(x_2s[0])
    y_p_post_2 = alpha_p_post_2 + np.dot(beta_p_post_2, x_2s)

    plt.plot(x_2s[0][idx_2], y_p_post_2[idx_2], 'D1', label=f'model order {order}')

