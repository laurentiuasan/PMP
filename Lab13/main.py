import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

"""
(1pt) Folosiţi metoda grid computing cu alte distribuţii a priori ca cea din curs.
    De exemplu, încercaţi cu prior = (grid<= 0.5).astype(int) , cu prior = abs(grid - 0.5)
    sau cu orice alt tip de distribuţie, oricât de ciudată.
    De asemenea, încercaţi să creşteţi cantitatea de date observate.
"""


def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    # prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return grid, posterior


# moneda aruncata de 16 ori si obtinem 6 steme
data = np.repeat([0, 1], (10, 6))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')

"""
(2pt) În codul folosit pentru estimarea lui π, fixaţi-l pe N şi rulaţi codul de mai multe ori. Veţi observa
    că rezultatele sunt diferite, deoarece folosim numere aleatoare.
    Puteţi estima care este legătura între numărul N de puncte şi eroare? Pentru o mai bună estimare,
    va trebui să modificaţi codul pentru a calcula eroarea ca funcţie de N.
    Puteţi astfel rula codul de mai multe ori cu acelaşi N (încercaţi, de exemplu N = 100, 1000 şi 10000),
    calcula media şi deviaţia standard a erorii, iar rezultatele le puteţi vizualiza cu funcţia plt.errorbar() 
    din matplotlib
"""


def estimate_value_of_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    # outside = np.invert(inside)
    # plt.figure(figsize=(8, 8))
    # plt.plot(x[inside], y[inside], 'b.')
    # plt.plot(x[outside], y[outside], 'r.')
    # plt.plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error: 4.3f}', alpha=0)
    # plt.axis('square')
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend(loc=1, frameon=True, framealpha=0.9)
    return error


print(estimate_value_of_pi(100))
print(estimate_value_of_pi(1000))
print((estimate_value_of_pi(10000)))
# eroarea scade cu cat numarul de puncte este mai mare

# media si deviata standard a erorii
errors = [estimate_value_of_pi(100), estimate_value_of_pi(1000), estimate_value_of_pi(10000)]
errors_mean = np.mean(errors)
errors_sd = np.std(errors)

plt.figure()
x = np.random.uniform(-1, 1, size=(2, 100))
y = np.exp(-x)

plt.figure()
plt.errorbar(x, y, errors_mean)
plt.show()

plt.figure()
plt.errorbar(x, y, errors_sd)
plt.show()

"""
3. (2pt) Modificaţi argumentul func din funcţia metropolis din curs folosind parametrii distribuţiei a
priori din Cursul 2 (pentru modelul beta-binomial) şi comparaţi cu metoda grid computing.
"""


def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5  # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


params = [(1, 1), (20, 20), (1, 4)]
for p1, p2 in params:
    func = stats.beta(p1, p2)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()

