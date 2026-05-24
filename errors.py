import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.integrate import trapezoid


def S_linear_analytic(x, N):
    """Аналитическая частичная сумма для f(x)=x"""
    s = 0
    for n in range(1, N + 1):
        s += ((-1) ** (n + 1) / n) * np.sin(n * x)
    return 2 * s


def S_abs_analytic(x, N):
    """Аналитическая частичная сумма для f(x)=|x|"""
    s = np.pi / 2
    for k in range(0, N):
        n = 2 * k + 1
        s -= (4 / (np.pi * n ** 2)) * np.cos(n * x)
    return s


def S_sign_analytic(x, N):
    """Аналитическая частичная сумма для sign(x)"""
    s = 0
    for k in range(0, N):
        n = 2 * k + 1
        s += (4 / (np.pi * n)) * np.sin(n * x)
    return s


def compute_coefficients_numerical(f, Nmax, M=1000):
    """Численное вычисление коэффициентов Фурье методом трапеций"""
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


def partial_sum_numerical(x, a, b, N):
    """Численная частичная сумма"""
    s = a[0] / 2
    for n in range(1, N + 1):
        s += a[n] * np.cos(n * x) + b[n] * np.sin(n * x)
    return s


x = np.linspace(-np.pi, np.pi, 500)
N = 10

# Для функции f(x)=x
a_num, b_num = compute_coefficients_numerical(lambda x: x, N, M=2000)
S_num = partial_sum_numerical(x, a_num, b_num, N)
S_analyt = S_linear_analytic(x, N)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, S_analyt, 'b-', label='Аналитическое разложение')
plt.plot(x, S_num, 'r--', label='Численное разложение')
plt.title('f(x)=x, N=10')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(S_analyt - S_num))
plt.title('Абсолютная разница')
plt.grid()
plt.show()


def compute_error(f, f_analytic, x, N_values, M=2000):
    """Вычисление ошибки для разных N"""
    errors = []
    for N in N_values:
        a_num, b_num = compute_coefficients_numerical(f, N, M)
        S_num = partial_sum_numerical(x, a_num, b_num, N)
        S_analyt = f_analytic(x, N)
        error = np.max(np.abs(S_analyt - S_num))
        errors.append(error)
    return errors

x_fine = np.linspace(-np.pi, np.pi, 1000)
N_values = [2, 4, 6, 8, 10, 15, 20, 30, 50]

# Ошибка для f(x)=|x|
errors_abs = compute_error(lambda x: np.abs(x), S_abs_analytic, x_fine, N_values)

plt.figure(figsize=(10, 6))
plt.loglog(N_values, errors_abs, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Число гармоник N')
plt.ylabel('Максимальная ошибка')
plt.title('Сходимость численного метода для f(x)=|x|')
plt.grid(True)
plt.show()


x_near = np.linspace(-0.3, 0.3, 2000)
N_values_gibbs = [5, 10, 20, 50, 100]

plt.figure(figsize=(12, 8))

# Аналитический подход
for N in N_values_gibbs:
    S = S_sign_analytic(x_near, N)
    plt.plot(x_near, S, label=f'N={N}')

plt.plot(x_near, np.sign(x_near), 'k--', linewidth=2, label='Исходная функция')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Эффект Гиббса в окрестности точки разрыва x=0')
plt.legend()
plt.grid()
plt.show()

# Исследование амплитуды выброса
max_values = []
for N in N_values_gibbs:
    S = S_sign_analytic(x_near, N)
    max_val = np.max(S[x_near > 0])
    max_values.append(max_val)
    print(f"N={N}: max = {max_val:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(N_values_gibbs, max_values, 'ro-', linewidth=2, markersize=8)
plt.axhline(y=1.18, color='k', linestyle='--', label='Теоретический предел ≈ 1.18')
plt.xlabel('Число гармоник N')
plt.ylabel('Максимальное значение выброса')
plt.title('Сходимость амплитуды выброса Гиббса')
plt.legend()
plt.grid()
plt.show()

n_values = np.arange(1, 101)
coeffs_linear = [2/n for n in n_values]
coeffs_abs = [4/(np.pi * n**2) for n in n_values if n % 2 == 1]
coeffs_sign = [4/(np.pi * n) for n in n_values if n % 2 == 1]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.loglog(n_values, coeffs_linear, 'b-')
plt.xlabel('n')
plt.ylabel('|b_n|')
plt.title('f(x)=x: |b_n| ~ 1/n')
plt.grid()

plt.subplot(1, 3, 2)
n_odd = np.arange(1, 101, 2)
plt.loglog(n_odd, coeffs_abs, 'r-')
plt.xlabel('n (нечётные)')
plt.ylabel('|a_n|')
plt.title('f(x)=|x|: |a_n| ~ 1/n²')
plt.grid()

plt.subplot(1, 3, 3)
plt.loglog(n_odd, coeffs_sign, 'g-')
plt.xlabel('n (нечётные)')
plt.ylabel('|b_n|')
plt.title('f(x)=sign(x): |b_n| ~ 1/n')
plt.grid()

plt.tight_layout()
plt.show()

# Оценка порядка сходимости
def estimate_order(errors, N_values):
    logN = np.log(N_values[-4:])
    logE = np.log(errors[-4:])
    coeffs = np.polyfit(logN, logE, 1)
    return -coeffs[0]

order_abs = estimate_order(errors_abs, N_values)
print(f"Порядок сходимости для |x|: {order_abs:.2f} (теория: 2.00)")
