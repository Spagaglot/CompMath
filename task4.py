#there is nothing to write a lot about
# 3 methods: Newton interpolation, natural cubic spline (custom tridiagonal solver), least squares

import numpy as np
import matplotlib.pyplot as plt

# data
years = np.array([1910,1920,1930,1940,1950,1960,1970,1980,1990,2000])
pop = np.array([92228496,106021537,123202624,132164569,151325798,179323175,203211926,226545805,248709873,281421906])

# evaluation grid
x_eval = np.linspace(years[0], 2010, 400)

# ---- 1) Newton polynomial interpolation ----
def divided_diffs(x, y):
    n = len(x)
    coef = y.astype(float).copy()
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def newton_eval(coef, x_data, x):
    n = len(coef)
    res = coef[-1]
    for k in range(n-2, -1, -1):
        res = res * (x - x_data[k]) + coef[k]
    return res

coef_newton = divided_diffs(years, pop)
newton_vals = newton_eval(coef_newton, years, x_eval)
val_2010_newton = float(newton_eval(coef_newton, years, 2010))

# ---- 2) Natural cubic spline with custom Thomas algorithm ----
def thomas_solve(a, b, c, d):
    # solve tridiagonal Ax=d where a=lower(1..n-1), b=diag(0..n-1), c=upper(0..n-2)
    n = len(b)
    cp = c.copy().astype(float)
    dp = d.copy().astype(float)
    bp = b.copy().astype(float)
    for i in range(1, n):
        m = a[i-1] / bp[i-1]
        bp[i] = bp[i] - m * cp[i-1]
        dp[i] = dp[i] - m * dp[i-1]
    x = np.zeros(n)
    x[-1] = dp[-1] / bp[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i+1]) / bp[i]
    return x

def natural_cubic_spline(x, y, xs):
    n = len(x)
    h = np.diff(x)
    # build system for second derivatives M (natural spline: M0=Mn=0)
    A = np.zeros(n-2)
    B = np.zeros(n-2)
    C = np.zeros(n-2)
    D = np.zeros(n-2)
    for i in range(1, n-1):
        A[i-1] = h[i-1]
        B[i-1] = 2*(h[i-1] + h[i])
        C[i-1] = h[i]
        D[i-1] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    if n>2:
        M_inner = thomas_solve(A, B, C, D)
        M = np.concatenate(([0.0], M_inner, [0.0]))
    else:
        M = np.array([0.0,0.0])
    # evaluate spline
    S = np.zeros_like(xs)
    for k, xx in enumerate(xs):
        # find interval
        if xx <= x[0]: i = 0
        elif xx >= x[-1]: i = n-2
        else:
            i = np.searchsorted(x, xx) - 1
        hi = x[i+1] - x[i]
        A1 = (x[i+1] - xx)/hi
        B1 = (xx - x[i]) / hi
        S[k] = A1*y[i] + B1*y[i+1] + ((A1**3 - A1)*M[i] + (B1**3 - B1)*M[i+1])*(hi**2)/6.0
    return S, M

spline_vals, M = natural_cubic_spline(years, pop, x_eval)
val_2010_spline = float(natural_cubic_spline(years, pop, np.array([2010]))[0])

# ---- 3) Least squares (degree 2) ----
coeffs_ls = np.polyfit(years, pop, 2)
poly_ls = np.poly1d(coeffs_ls)
ls_vals = poly_ls(x_eval)
val_2010_ls = float(poly_ls(2010))


print('Newton 2010:', int(round(val_2010_newton)))
print('Spline 2010:', int(round(val_2010_spline)))
print('Least squares 2010:', int(round(val_2010_ls)))

plt.figure(figsize=(8,4))
plt.plot(years, pop, 'o', label='data')
plt.plot(x_eval, newton_vals, label='Newton')
plt.title('Newton interpolation')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,4))
plt.plot(years, pop, 'o', label='data')
plt.plot(x_eval, spline_vals, label='Cubic spline')
plt.title('Natural cubic spline')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,4))
plt.plot(years, pop, 'o', label='data')
plt.plot(x_eval, ls_vals, label='Least squares deg2')
plt.title('Least squares (degree 2)')
plt.legend()
plt.tight_layout()

plt.show()
