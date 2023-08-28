# -*- coding: UTF-8 -*-

"""
Macroeconomics II - Practical Session 1
Author: Matheus Franciscão
"""

# Imports
import numpy as np
from gensys import gensys
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


beta = 0.99
sigma = 1
kappa = 1.2
phi = 1.5
rho = 0.05
rho_m = 0.90

g0 = np.array([[1, 1/sigma, 0, 0],[0, beta, 0, 0], [0, -phi, 1, -1], [0, 0, 0, 1]])

g1 = np.array([[1, 0, 1/sigma, 0], [-kappa, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, rho_m]])

Psi = np.array([[0], [0], [0], [1]])

Pi = np.array([[-1, -1/sigma], [-beta, 0], [0, 0], [0, 0]])

Const = np.array([[-rho/sigma], [0], [rho], [0]])

G1, C, impact, eu = gensys(g0, g1, Const, Psi, Pi, div=1, realsmall=0.000001)

print(eu)

if eu == [1, 1]:
    T = 30
    series = np.zeros((T + 1, 4))
    series[0:1, :] = impact.T
    for t in range(1, T + 1):
        series[t, :] = G1 @ series[t - 1, :]
    figure = plt.figure()
    x_axis = range(T + 1)
    titles = ["y_t", "π_t", "r_t", "u_t"]
    for i in range(4):
        chart = figure.add_subplot(3, 2, i+1)
        chart.plot(x_axis, series[:,i])
        chart.set_title(titles[i])
    plt.tight_layout()
    plt.draw()
    plt.show()


