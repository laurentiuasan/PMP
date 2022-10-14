"""
depozit cu alarma
p(cutremur) = 0.05%
p(incendiu) = 1%
p(incendiu|cutremur) = 3%

p(accidental) = 0.01%
p(accidental | cutremur) = 2%
p(alarma.incendiu) = 95%
p(alarma.incendiu| cutremur) = 98%
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


if __name__ == '__main__':
    model = pm.Model()

    with model:
        incendiu = pm.Bernoulli('I', 0.01)
        cutremur = pm.Bernoulli('C', 0.0005)
        alarma = pm.Bernoulli('A', 0.95)

        alarma_c = pm.Deterministic('IC', pm.math.switch(incendiu, pm.math.switch(cutremur, 0.98, 0.95), pm.math.switch(cutremur, 0.02, 0.0001)))
        alarma_c = pm.Bernoulli('AC', p=alarma_c)

        trace = pm.sample(10000)

    dictionary = {
        'incendiu': trace['I'].tolist(),
        'cutremur': trace['C'].tolist()
    }
    df = pd.DataFrame(dictionary)

    p_cutremur_alarma = df[((df['cutremur'] == 1) & (df['incendiu'] == 1))].shape[0] / df[df['incendiu'] == 1].shape[0]
    print(p_cutremur_alarma)

    az.plot_posterior(trace)
    plt.show()