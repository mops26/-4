import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq, ifft

def load_tortilla_data(file_path):
    """Загрузка данных о ценах на тортильи"""
    df = pd.read_csv('tortilla_prices.csv')
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {df.columns.tolist()}")
    return df


def prepare_time_series(df, state=None, city=None):
    """
    Подготовка временного ряда: группировка по датам
    state - фильтр по штату (например, 'Mexico City')
    city - фильтр по городу
    """
    if state:
        df = df[df['State'].astype(str).str.contains(state, case=False, na=False)]
        print(f"После фильтрации по штату '{state}': {len(df)} записей")

    if city:
        df = df[df['City'].astype(str).str.contains(city, case=False, na=False)]
        print(f"После фильтрации по городу '{city}': {len(df)} записей")

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    df_grouped = df.groupby('Date')['Price per kilogram'].mean().reset_index()
    df_grouped = df_grouped.sort_values('Date')

    df_grouped.set_index('Date', inplace=True)
    df_monthly = df_grouped.resample('ME').mean().dropna().reset_index()

    print(f"Сформирован временной ряд из {len(df_monthly)} точек (помесячно)")

    return df_monthly['Date'].values, df_monthly['Price per kilogram'].values


def preprocess_data(values):
    """Удаление линейного тренда для фокуса на колебаниях"""
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    trend = np.polyval(coeffs, x)
    return values - trend, coeffs


def compute_spectrum(signal, sample_rate=1.0):
    """Вычисление спектра сигнала"""
    n = len(signal)
    fft_vals = fft(signal)
    frequencies = fftfreq(n, d=1 / sample_rate)[:n // 2]
    amplitudes = np.abs(fft_vals)[:n // 2] / n
    return frequencies, amplitudes, fft_vals


def filter_signal(fft_vals, frequencies, keep_harmonics_count):
    """Оставляет только keep_harmonics_count самых сильных гармоник"""
    amplitudes = np.abs(fft_vals[:len(frequencies)])
    indices = np.argsort(amplitudes)[::-1][:keep_harmonics_count]

    fft_filtered = np.zeros_like(fft_vals, dtype=complex)
    for idx in indices:
        fft_filtered[idx] = fft_vals[idx]
        # Для отрицательных частот (симметрия)
        if idx != 0:
            fft_filtered[-idx] = fft_vals[-idx]

    return fft_filtered


def reconstruct_signal(fft_filtered):
    """Восстанавливает сигнал из отфильтрованного спектра"""
    return np.real(ifft(fft_filtered))


# ============================================================
# 3. ОЦЕНКА КАЧЕСТВА
# ============================================================

def compute_metrics(original, reconstructed):
    """Оценка качества аппроксимации"""
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))

    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}


# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_analysis(dates, original, signal, frequencies, amplitudes, reconstructed, n_harmonics, metrics):
    """Построение всех графиков"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Исходный временной ряд
    axes[0, 0].plot(dates, original, 'b-', linewidth=1)
    axes[0, 0].set_title('Цены на тортильи в Мексике (исходные данные)')
    axes[0, 0].set_xlabel('Дата')
    axes[0, 0].set_ylabel('Цена за кг (песо)')
    axes[0, 0].grid(True)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

    # 2. Спектр сигнала (после удаления тренда)
    axes[0, 1].stem(frequencies[:50], amplitudes[:50], basefmt=' ')
    axes[0, 1].set_title('Спектр сигнала (первые 50 частот)')
    axes[0, 1].set_xlabel('Частота (1/месяц)')
    axes[0, 1].set_ylabel('Амплитуда')
    axes[0, 1].grid(True)

    # 3. Сравнение исходного и восстановленного сигнала
    axes[1, 0].plot(dates, original, 'b-', label='Исходные цены', linewidth=1, alpha=0.7)
    axes[1, 0].plot(dates, reconstructed, 'r-', label=f'Восстановленные (N={n_harmonics})', linewidth=1)
    axes[1, 0].set_title('Сравнение исходных и восстановленных цен')
    axes[1, 0].set_xlabel('Дата')
    axes[1, 0].set_ylabel('Цена за кг (песо)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # 4. Ошибка восстановления
    error = original - reconstructed
    axes[1, 1].plot(dates, error, 'g-', linewidth=1)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Ошибка восстановления\nMSE = {metrics["MSE"]:.4f}, R² = {metrics["R²"]:.4f}')
    axes[1, 1].set_xlabel('Дата')
    axes[1, 1].set_ylabel('Ошибка (песо)')
    axes[1, 1].grid(True)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('tortilla_analysis.png', dpi=150)
    plt.show()


def plot_harmonics_impact(dates, original, signal, frequencies, fft_vals, max_harmonics=20):
    """Исследование влияния числа гармоник на качество восстановления"""
    errors = {'MSE': [], 'R2': []}
    harmonics_counts = range(1, max_harmonics + 1)

    for k in harmonics_counts:
        fft_filtered = filter_signal(fft_vals, frequencies, k)
        reconstructed_detrend = reconstruct_signal(fft_filtered)

        # Добавляем обратно тренд
        x = np.arange(len(original))
        coeffs = np.polyfit(x, original, 1)
        trend = np.polyval(coeffs, x)
        reconstructed = reconstructed_detrend + trend

        metrics = compute_metrics(original, reconstructed)
        errors['MSE'].append(metrics['MSE'])
        errors['R2'].append(metrics['R²'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(harmonics_counts, errors['MSE'], 'bo-')
    axes[0].set_xlabel('Число гармоник')
    axes[0].set_ylabel('MSE (среднеквадратичная ошибка)')
    axes[0].set_title('Влияние числа гармоник на MSE')
    axes[0].grid(True)

    axes[1].plot(harmonics_counts, errors['R2'], 'ro-')
    axes[1].set_xlabel('Число гармоник')
    axes[1].set_ylabel('R² (коэффициент детерминации)')
    axes[1].set_title('Влияние числа гармоник на R²')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('tortilla_harmonics_impact.png', dpi=150)
    plt.show()

    return errors


# ============================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main(file_path, state=None, city=None):
    """Основная функция анализа"""

    # 1. Загрузка и подготовка данных
    df = load_tortilla_data(file_path)
    dates, prices = prepare_time_series(df, state=state, city=city)

    # 2. Удаление тренда
    signal_detrend, trend_coeffs = preprocess_data(prices)
    x = np.arange(len(prices))
    trend = np.polyval(trend_coeffs, x)

    # 3. Спектральный анализ
    sample_rate = 1.0  # 1 наблюдение в месяц
    frequencies, amplitudes, fft_vals = compute_spectrum(signal_detrend, sample_rate)

    # 4. Вывод доминирующих частот
    n_top = 5
    top_indices = np.argsort(amplitudes)[::-1][:n_top]
    print("\nДоминирующие частоты и периоды:")
    for idx in top_indices:
        freq = frequencies[idx]
        period = 1 / freq if freq > 0 else np.inf
        print(f"  Частота: {freq:.4f} 1/мес, Период: {period:.1f} мес, Амплитуда: {amplitudes[idx]:.4f}")

    # 5. Восстановление по 5 гармоникам
    n_harmonics = 5
    fft_filtered = filter_signal(fft_vals, frequencies, n_harmonics)
    reconstructed_detrend = reconstruct_signal(fft_filtered)
    reconstructed = reconstructed_detrend + trend

    # 6. Оценка качества
    metrics = compute_metrics(prices, reconstructed)
    print(f"\nКачество восстановления по {n_harmonics} гармоникам:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")

    # 7. Построение графиков
    plot_analysis(dates, prices, signal_detrend, frequencies, amplitudes, reconstructed, n_harmonics, metrics)

    # 8. Исследование влияния числа гармоник
    plot_harmonics_impact(dates, prices, signal_detrend, frequencies, fft_vals, max_harmonics=15)

    return df, dates, prices, reconstructed


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    FILE_PATH = "tortilla_prices.csv"

    # Можно фильтровать по штату, например 'Mexico City'
    # или оставить None для всех данных
    df, dates, prices, reconstructed = main(FILE_PATH, state=None, city=None)
