import numpy as np

def griewank(X):
    X = np.asarray(X)
    j = np.sqrt(np.arange(1, 4))
    term1 = np.sum(X**2, axis=-1) / 4000
    term2 = np.prod(np.cos(X / j), axis=-1)
    return term1 - term2 + 1

def ackley(X):
    X = np.asarray(X)
    d = X.shape[-1]
    sum_sq = np.sum(X**2, axis=-1)
    cos_sum = np.sum(np.cos(2 * np.pi * X), axis=-1)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(cos_sum / d) + 20 + np.e

def rastrigin(X):
    X = np.asarray(X)
    A = 10
    D = X.shape[-1]
    return A * D + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)

def rosenbrock(X):
    X = np.asarray(X)
    return np.sum(100 * (X[..., 1:] - X[..., :-1]**2)**2 + (1 - X[..., :-1])**2, axis=-1)