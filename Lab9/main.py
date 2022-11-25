import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd
import theano
theano.config.compute_test_value = "ignore"


if __name__ == '__main__':
    data = pd.read_csv('Admission.csv')
    data.head()
    df = data.query("Admission == ('0', '1')")
    admission = pd.Categorical(df['Admission']).codes
    x_n = ['GRE', 'GPA']
    x_1 = df['GRE'].values
    x_2 = df['GPA'].values

    model = pm.Model()
    with model:
        alpha = pm.Normal("alpha", mu=0, sd=10)
        beta = pm.Normal("beta", mu=0, sd=2, shape=len(x_n))
        mu = alpha + pm.math.dot(x_1, beta[0]) + pm.math.dot(x_2, beta[1])
        teta = pm.Deterministic("teta", (1 / pm.math.exp(-mu)))

        admission_l = pm.Bernoulli('admissionl', p=teta, observed=admission)
        idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)
