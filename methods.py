import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.integrate import trapezoid
import time


def compute_coefficients_direct(f, Nmax, M=1000):
    """Прямое вычисление коэффициентов Фурье методом трапеций"""
    x = np.linspace(-np.pi, np.pi, M)
    h = 2 * np.pi / M
    y = f(x)

    a = np.zeros(Nmax + 1)
    b = np.zeros(Nmax + 1)

    a[0] = (1 / np.pi) * trapezoid(y, dx=h)

    for n in range(1, Nmax + 1):
        a[n] = (1 / np.pi) * trapezoid(y * np.cos(n * x), dx=h)
        b[n] = (1 / np.pi) * trapezoid(y * np.sin(n * x), dx=h)

    return a, b



def compute_coefficients_fft(f, Nmax, M=1000):
    """Вычисление коэффициентов Фурье через БПФ"""
    x = np.linspace(-np.pi, np.pi, M, endpoint=False)
    y = f(x)

    F = fft(y) / M
    a = np.zeros(Nmax + 1)
    b = np.zeros(Nmax + 1)

    a[0] = F[0].real * 2
    for n in range(1, Nmax + 1):
        if n < M // 2:
            a[n] = 2 * F[n].real
            b[n] = -2 * F[n].imag
        else:
            a[n] = 0
            b[n] = 0
    return a, b



def partial_sum(x, a, b, N):
    """Частичная сумма ряда Фурье"""
    s = a[0] / 2
    for n in range(1, N + 1):
        s += a[n] * np.cos(n * x) + b[n] * np.sin(n * x)
    return s


def f_sign(x):
    if isinstance(x, (int, float)):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    else:
        return np.where(x > 0, 1, np.where(x < 0, -1, 0))


def f_abs(x):
    return np.abs(x)


def f_linear(x):
    return x


x_plot = np.linspace(-np.pi, np.pi, 1000)
N_values = [1, 3, 5, 10, 20]

plt.figure(figsize=(10, 6))
a, b = compute_coefficients_direct(f_sign, max(N_values), M=1000)
plt.plot(x_plot, f_sign(x_plot), 'k--', label='Исходная функция', linewidth=2)

for N in N_values:
    S = partial_sum(x_plot, a, b, N)
    plt.plot(x_plot, S, label=f'N={N}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Аппроксимация функции sgn(x) частичными суммами ряда Фурье')
plt.legend()
plt.grid()
plt.savefig('fourier_convergence.png', dpi=150)
plt.show()


M = 5000
Nmax = 100

start = time.time()
a_direct, b_direct = compute_coefficients_direct(f_abs, Nmax, M)
time_direct = time.time() - start

start = time.time()
a_fft, b_fft = compute_coefficients_fft(f_abs, Nmax, M)
time_fft = time.time() - start

print(f"Прямой метод: {time_direct:.3f} с")
print(f"БПФ: {time_fft:.3f} с")
print(f"Ускорение: {time_direct / time_fft:.1f} раз")


a, b = compute_coefficients_direct(f_linear, 50, M=2000)
S = partial_sum(x_plot, a, b, 10)
exact = f_linear(x_plot)
error = np.abs(S - exact)

plt.figure(figsize=(10, 5))
plt.plot(x_plot, error)
plt.xlabel('x')
plt.ylabel('Погрешность')
plt.title('Абсолютная погрешность аппроксимации f(x)=x при N=10')
plt.grid()
plt.savefig('fourier_error.png', dpi=150)
plt.show()
