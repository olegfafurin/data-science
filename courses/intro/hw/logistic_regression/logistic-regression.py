import numpy as np, pandas as pd
from sklearn import metrics


def evaluate(omega, x, y, ind):
    omega_1 = omega[0]
    omega_2 = omega[1]
    s = 0
    for i in range(len(x)):
        s += y[i] * x[i, ind] * (1 - 1 / (1 + np.exp(-y[i] * (omega_1 * x[i, 0] + omega_2 * x[i, 1]))))
    s /= len(x)
    return s


def make_step(omega, step, x, y, c):
    delta_1 = evaluate(omega, x, y, 0) * step
    delta_2 = evaluate(omega, x, y, 1) * step
    delta_1 -= step * c * omega[0]
    delta_2 -= step * c * omega[1]
    return np.array([omega[0] + delta_1, omega[1] + delta_2])


def grad_descent(step, omega, x, y, c):
    old_omega = np.array([1.0, 1.0])
    for _ in range(10000):
        if np.sqrt((sum(np.square(omega - old_omega)))) > pow(10.0, -5):
            old_omega = np.copy(omega)
            omega = make_step(omega, step, x, y, c)
        else:
            break
    return omega


sigma = lambda x, omega: 1 / (1 + np.exp(-omega[0] * x[0] - omega[1] * x[1]))

data = pd.read_csv("data-logistic.csv", header=None)
q = np.array(data[[1, 2]])

coef = grad_descent(0.1, np.array([0, 0]), np.array(data[[1, 2]]), data[0], 0)
reg_coef = grad_descent(0.1, np.array([0, 0]), np.array(data[[1, 2]]), data[0], 10)

prob = [sigma(q[i], coef) for i in range(len(data))]
reg_prob = [sigma(q[i], reg_coef) for i in range(len(data))]

print(metrics.roc_auc_score(data[0], prob))
print(metrics.roc_auc_score(data[0], reg_prob))
