import numpy as np
import matplotlib.pyplot as plt

def S_linear(x, N):
    """Частичная сумма ряда Фурье для f(x)=x"""
    s = 0
    for n in range(1, N+1):
        s += ((-1)**(n+1) / n) * np.sin(n * x)
    return 2 * s

def S_abs(x, N):
    """Частичная сумма ряда Фурье для f(x)=|x|"""
    s = np.pi/2
    for k in range(0, N):
        n = 2*k + 1  # только нечётные гармоники
        s -= (4 / (np.pi * n**2)) * np.cos(n * x)
    return s


def S_sign(x, N):
    """Частичная сумма ряда Фурье для sign(x)"""
    s = 0
    for k in range(0, N):
        n = 2*k + 1
        s += (4 / (np.pi * n)) * np.sin(n * x)
    return s


x = np.linspace(-np.pi, np.pi, 1000)


N_values = [1, 3, 5, 10, 20]

plt.figure(figsize=(12, 10))

# График 1: f(x) = x
plt.subplot(3, 1, 1)
plt.plot(x, x, 'k--', linewidth=2, label='Исходная функция f(x)=x')
for N in N_values:
    plt.plot(x, S_linear(x, N), label=f'N={N}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Разложение функции f(x)=x в ряд Фурье')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, np.abs(x), 'k--', linewidth=2, label='Исходная функция f(x)=|x|')
for N in N_values:
    plt.plot(x, S_abs(x, N), label=f'N={N}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Разложение функции f(x)=|x| в ряд Фурье')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, np.sign(x), 'k--', linewidth=2, label='Исходная функция sign(x)')
for N in N_values:
    plt.plot(x, S_sign(x, N), label=f'N={N}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Разложение функции sign(x) в ряд Фурье')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fourier_sums.png', dpi=150)
plt.show()
