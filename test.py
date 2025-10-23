import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu
import matplotlib.pyplot as plt
from time import time


def precondition(A):
    return spilu(A.tocsc())

def norm_1(u):
    return np.max(np.abs(u))

def norm_2(u):
    return np.sum(np.abs(u))

def norm_3(u):
    return np.sqrt(np.dot(u,u))


### BiCGStab с счетчиком итераций

def BiCGStab(A_, f_, x_0, e):
    A = sparse.csr_matrix(A_)
    f = f_.copy()
    x = x_0.copy()
    M = precondition(A)
    residuals_1, residuals_2, residuals_3 = [], [], []

    r_0 = f - A.dot(x)
    r = r_0.copy()
    rho_prev = alpha = omega = 1.0
    v = p = np.zeros_like(x)

    iteration = 0

    while min(norm_1(f - A.dot(x)), norm_2(f - A.dot(x)), norm_3(f - A.dot(x))) > e:
        iteration += 1

        rho = np.dot(r_0, r)
        if abs(rho) < e:
            raise RuntimeError("BiCGStab cannot solve the system")
        if rho_prev == 1.0:
            p = r.copy()
        else:
            beta = (rho/rho_prev)*(alpha/omega)
            p = r + beta*(p - omega*v)

        p_ = M.solve(p)
        v = A.dot(p_)
        alpha = rho / np.dot(r_0, v)
        s = r - alpha * v

        if norm_3(s)/norm_3(r_0) < e:
            x += alpha * p_
            residuals_1.append(norm_1(f - A.dot(x)))
            residuals_2.append(norm_2(f - A.dot(x)))
            residuals_3.append(norm_3(f - A.dot(x)))
            break

        s_ = M.solve(s)
        t = A.dot(s_)
        omega = np.dot(t, s) / np.dot(t, t)
        x += alpha * p_ + omega * s_
        r = s - omega * t

        rho_prev = rho
        residuals_1.append(norm_1(f - A.dot(x)))
        residuals_2.append(norm_2(f - A.dot(x)))
        residuals_3.append(norm_3(f - A.dot(x)))

    print(f"Количество итераций: {iteration}")

    return x, residuals_1, residuals_2, residuals_3

### геним большую матрицу
def make_sparse_matrix(n, density=5e-4, seed=42):
    rng = np.random.default_rng(seed)
    S = sparse.random(n, n, density=density, format='csr', data_rvs=rng.standard_normal)
    diag = sparse.diags(np.abs(S).sum(axis=1).A.ravel() + 1.0)  # диагонально-доминантная
    A = S + diag
    return A

### параметры
n = 10000
A = make_sparse_matrix(n)
x_true = np.random.randn(n)
b = A.dot(x_true)

x0 = np.zeros(n)
tol = 1e-8


t0 = time()
x, res1, res2, res3 = BiCGStab(A, b, x0, tol)
t1 = time()
print(f"BiCGStab завершен: время {t1-t0:.2f} с, последняя невязка = {np.linalg.norm(b - A.dot(x)):.3e}")

### график
residuals = np.array([res1, res2, res3])
iterations = np.arange(1, residuals.shape[1]+1)

plt.figure(figsize=[14, 7])
plt.title("BiCGStab method", fontsize=20)
plt.yscale("log")
plt.xlabel("Итерации", fontsize=16)
plt.ylabel("Норма невязки (логарифмическая шкала)", fontsize=16)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.plot(iterations, residuals[0], label="norm 1")
plt.plot(iterations, residuals[1], label="norm 2")
plt.plot(iterations, residuals[2], label="norm 3")
plt.legend()
plt.show()


