import numpy as np
import matplotlib.pyplot as plt

#определим функции
functions = [
    (lambda x: np.sin(x**2), lambda x: 2*x*np.cos(x**2), 'sin(x^2)'),
    (lambda x: np.cos(np.sin(x)), lambda x: -np.sin(np.sin(x))*np.cos(x), 'cos(sin(x))'),
    (lambda x: np.exp(np.sin(np.cos(x))), lambda x: -np.exp(np.sin(np.cos(x)))*np.cos(np.cos(x))*np.sin(x), 'exp(sin(cos(x)))'),
    (lambda x: np.log(x+3), lambda x: 1/(x+3), 'ln(x+3)'),
    (lambda x: np.sqrt(x+3), lambda x: 1/(2*np.sqrt(x+3)), '(x+3)^0.5')
]

#точка, в которой считаем производную
x0 = 1.0

#массив шагов h_n = 2 / 2^n, n = 1..21
n_values = np.arange(1, 22)
h_values = 2 / 2**n_values

#определим 5 методов численного дифференцирования

""" прямой метод дифференцирования/погрешность порядка O(h)"""
def forward(f, x, h):
    return (f(x+h) - f(x)) / h

""" обратный метод дифференцирования/погрешность порядка O(h)"""
def backward(f, x, h):
    return (f(x) - f(x-h)) / h

"""центральная разность/погрешность порядка O(h^2)"""
def central(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

"""четырёхточеченая формула/погрешность порядка O(h^4)"""
def four_point(f, x, h):
    return 4/3 * central(f, x, h) - 1/3 * (f(x+2*h) - f(x-2*h)) / (4*h)

"""пятиточечная формула/погрешность порядка O(h^6)"""
def five_point(f, x, h):
    return (3/2 * central(f, x, h)
            - 3/5 * (f(x+2*h) - f(x-2*h)) / (4*h)
            + 1/10 * (f(x+3*h) - f(x-3*h)) / (6*h))

methods = [
    (forward, 'Forward'),
    (backward, 'Backward'),
    (central, 'Central'),
    (four_point, '4-point'),
    (five_point, '5-point')
]

#cтроим 5 графиков
for f, f_prime, fname in functions:
    plt.figure(figsize=(8,6))
    for method, mname in methods:
        errors = [abs(method(f, x0, h) - f_prime(x0)) for h in h_values]
        plt.loglog(h_values, errors, label=mname, marker='o')

    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error vs Step Size for {fname}')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()
