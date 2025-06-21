import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, Boltzmann

# Константы
c = 3e8  # скорость света, м/с
freq = 50e6  # частота несущей, Гц
wavelength = c / freq  # длина волны, м

# Параметры по умолчанию из задания
default_params = {
    'P_t': 10,  # мощность передатчика, Вт
    'G_t': 10 ** (3 / 10),  # коэффициент усиления передающей антенны (3 dB)
    'G_r': 10 ** (5 / 10),  # коэффициент усиления приемной антенны (5 dB)
    'R': 100e6,  # расстояние, м (100,000 км)
    'L': 10 ** (1 / 10),  # потери в атмосфере (1 dB)
    'T_n': 100,  # шумовая температура приемника, K
    'FEC_rate': 3 / 4,  # скорость кода коррекции ошибок
    'EbN0_req': 10 ** (3 / 10)  # требуемое Eb/N0 для BER=1e-3 (3 dB)
}


def calculate_bit_rate(params):
    """Рассчитывает максимальную скорость передачи данных для заданных параметров"""
    # Расчет мощности на входе приемника (уравнение дальности связи)
    P_r = (params['P_t'] * params['G_t'] * params['G_r'] * wavelength ** 2) / \
          ((4 * pi * params['R']) ** 2 * params['L'])

    # Расчет спектральной плотности шума
    N0 = Boltzmann * params['T_n']

    # Расчет отношения сигнал-шум на бит (Eb/N0)
    EbN0 = (P_r / N0)

    # Расчет максимальной скорости передачи (из требования Eb/N0)
    bit_rate = EbN0 / params['EbN0_req']

    # Расчет информационной скорости с учетом FEC
    info_rate = bit_rate * params['FEC_rate']

    return bit_rate, info_rate


def plot_orbit_distance_effect():
    """График зависимости скорости передачи от расстояния"""
    distances = np.linspace(50e6, 200e6, 100)  # от 50,000 до 200,000 км
    bit_rates = []
    info_rates = []

    for dist in distances:
        params = default_params.copy()
        params['R'] = dist
        br, ir = calculate_bit_rate(params)
        bit_rates.append(br)
        info_rates.append(ir)

    plt.figure(figsize=(10, 6))
    plt.plot(distances / 1e6, np.array(bit_rates) / 1e3, label='Скорость передачи (кбит/с)')
    plt.plot(distances / 1e6, np.array(info_rates) / 1e3, label='Информационная скорость (кбит/с)')
    plt.axhline(630, color='r', linestyle='--', label='Целевая скорость 630 кбит/с')
    plt.xlabel('Расстояние (км)')
    plt.ylabel('Скорость (кбит/с)')
    plt.title('Зависимость скорости передачи от расстояния')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_transmitter_power_effect():
    """График зависимости скорости передачи от мощности передатчика"""
    powers = np.linspace(1, 20, 100)  # от 1 до 20 Вт
    bit_rates = []
    info_rates = []

    for power in powers:
        params = default_params.copy()
        params['P_t'] = power
        br, ir = calculate_bit_rate(params)
        bit_rates.append(br)
        info_rates.append(ir)

    plt.figure(figsize=(10, 6))
    plt.plot(powers, np.array(bit_rates) / 1e3, label='Скорость передачи (кбит/с)')
    plt.plot(powers, np.array(info_rates) / 1e3, label='Информационная скорость (кбит/с)')
    plt.axhline(630, color='r', linestyle='--', label='Целевая скорость 630 кбит/с')
    plt.xlabel('Мощность передатчика (Вт)')
    plt.ylabel('Скорость (кбит/с)')
    plt.title('Зависимость скорости передачи от мощности передатчика')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_noise_temperature_effect():
    """График зависимости скорости передачи от шумовой температуры"""
    temperatures = np.linspace(50, 200, 100)  # от 50K до 200K
    bit_rates = []
    info_rates = []

    for temp in temperatures:
        params = default_params.copy()
        params['T_n'] = temp
        br, ir = calculate_bit_rate(params)
        bit_rates.append(br)
        info_rates.append(ir)

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, np.array(bit_rates) / 1e3, label='Скорость передачи (кбит/с)')
    plt.plot(temperatures, np.array(info_rates) / 1e3, label='Информационная скорость (кбит/с)')
    plt.axhline(630, color='r', linestyle='--', label='Целевая скорость 630 кбит/с')
    plt.xlabel('Шумовая температура приемника (K)')
    plt.ylabel('Скорость (кбит/с)')
    plt.title('Зависимость скорости передачи от шумовой температуры')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_atmosphere_loss_effect():
    """График зависимости скорости передачи от потерь в атмосфере"""
    losses_db = np.linspace(0, 5, 100)  # от 0 до 5 dB
    bit_rates = []
    info_rates = []

    for loss in losses_db:
        params = default_params.copy()
        params['L'] = 10 ** (loss / 10)
        br, ir = calculate_bit_rate(params)
        bit_rates.append(br)
        info_rates.append(ir)

    plt.figure(figsize=(10, 6))
    plt.plot(losses_db, np.array(bit_rates) / 1e3, label='Скорость передачи (кбит/с)')
    plt.plot(losses_db, np.array(info_rates) / 1e3, label='Информационная скорость (кбит/с)')
    plt.axhline(630, color='r', linestyle='--', label='Целевая скорость 630 кбит/с')
    plt.xlabel('Потери в атмосфере (dB)')
    plt.ylabel('Скорость (кбит/с)')
    plt.title('Зависимость скорости передачи от потерь в атмосфере')
    plt.grid(True)
    plt.legend()
    plt.show()


# Запуск построения графиков
plot_orbit_distance_effect()
plot_transmitter_power_effect()
plot_noise_temperature_effect()
plot_atmosphere_loss_effect()