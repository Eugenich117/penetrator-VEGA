import numpy as np
import matplotlib.pyplot as plt

# Константы
g_B = 8.871  # Ускорение свободного падения, м/с²
pi = np.pi  # Математическая константа

# Фиксированные параметры по условию
h_fixed = 0.1       # Шаг винта (м)
v_sigma_fixed = 0.1035  # Радиус винта (м)
N_pot_fixed = 4.0   # Количество винтов

# Базовые значения для других параметров
n_base = 82     # Базовый параметр n (частота вращения)
rho_sv_base = 1.225  # Плотность воздуха при стандартных условиях (кг/м³)

# Диапазоны значений для варьирования
rho_sv_values = np.arange(0.5, 80.1, 1.0)  # Плотность среды (кг/м³)
v_sigma_values = np.arange(0.05, 0.5, 0.01)  # Радиус винта (м)
h_values = np.arange(0.01, 1.0, 0.01)       # Шаг винта (м)
n_values = np.arange(0, 100, 1)             # Частота вращения (Гц)
N_pot_values = np.arange(1, 10, 1)          # Количество винтов

def calculate_takeoff_mass(rho_sv, v_sigma, h, n, N_pot):
    """Расчёт взлётной массы с проверкой входных параметров"""
    numerator = v_sigma**2 * h**2 * n**2 * N_pot
    if numerator <= 0:
        return 0.0
    return pi * rho_sv * (numerator / g_B)

def calculate_n(M, rho_sv, v_sigma, h, N_pot):
    """Расчёт частоты вращения с проверкой входных параметров"""
    denominator = pi * rho_sv * v_sigma**2 * h**2 * N_pot
    if denominator <= 0:
        return 0.0
    return np.sqrt((M * g_B) / denominator)

# 1. Зависимость от плотности среды (фиксируем все кроме rho_sv)
M_rho = [calculate_takeoff_mass(rho, v_sigma_fixed, h_fixed, n_base, N_pot_fixed)
         for rho in rho_sv_values]

# 2. Зависимость от радиуса винта (фиксируем все кроме v_sigma)
M_v = [calculate_takeoff_mass(rho_sv_base, v, h_fixed, n_base, N_pot_fixed)
       for v in v_sigma_values]

# 3. Зависимость от шага винта (фиксируем все кроме h)
M_h = [calculate_takeoff_mass(rho_sv_base, v_sigma_fixed, h, n_base, N_pot_fixed)
       for h in h_values]

# 4. Зависимость от частоты вращения (фиксируем все кроме n)
M_n = [calculate_takeoff_mass(rho_sv_base, v_sigma_fixed, h_fixed, n, N_pot_fixed)
       for n in n_values]

# 5. Зависимость от количества винтов (фиксируем все кроме N_pot)
M_N_pot = [calculate_takeoff_mass(rho_sv_base, v_sigma_fixed, h_fixed, n_base, N_pot)
           for N_pot in N_pot_values]

# 6. Зависимость частоты вращения от плотности среды (фиксируем все кроме rho_sv)
M_target = 15.0  # Целевая взлётная масса (кг)
n_from_rho = [calculate_n(M_target, rho, v_sigma_fixed, h_fixed, N_pot_fixed)
              for rho in rho_sv_values]

# Построение графиков
def plot_graph(x, y, xlabel, ylabel, title, color, highlight_point=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color=color)
    if highlight_point:
        plt.scatter([highlight_point[0]], [highlight_point[1]], color='red', s=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Графики зависимостей
plot_graph(rho_sv_values, M_rho,
           'Плотность среды (ρ_СВ, кг/м³)', 'Взлётная масса (кг)',
           'Зависимость взлётной массы от плотности среды', 'blue')

plot_graph(v_sigma_values, M_v,
           'Радиус винта (м)', 'Взлётная масса (кг)',
           'Зависимость взлётной массы от радиуса винта', 'green')

plot_graph(h_values, M_h,
           'Шаг винта (h, м)', 'Взлётная масса (кг)',
           'Зависимость взлётной массы от шага винта', 'red')

plot_graph(n_values, M_n,
           'Частота вращения ротора (Гц)', 'Взлётная масса (кг)',
           'Зависимость взлётной массы от частоты вращения', 'purple')

plot_graph(N_pot_values, M_N_pot,
           'Количество винтов', 'Взлётная масса (кг)',
           'Зависимость взлётной массы от количества винтов', 'orange')

# Специальный график для частоты вращения
n_target = 4900 / 60  # 4900 об/мин → 81.6667 Гц
rho_target = (M_target * g_B) / (pi * v_sigma_fixed**2 * h_fixed**2 * N_pot_fixed * n_target**2)
plot_graph(rho_sv_values, n_from_rho,
           'Плотность среды (ρ_СВ, кг/м³)', 'Частота вращения (n, Гц)',
           'Зависимость частоты вращения от плотности среды в состоянии зависания', 'cyan',
           highlight_point=(rho_target, n_target))


# Константы
g_B = 8.871  # Ускорение свободного падения на Венере, м/с²
pi = np.pi

# Фиксированные параметры (скорректированные для реалистичности)
h_fixed = 0.1        # Шаг винта (м) - увеличен
v_sigma_fixed = 0.1035  # Радиус винта (м) - увеличен до 1 метра
N_pot_fixed = 4      # Количество винтов
rho_sv_venus = 1.225  # Плотность атмосферы Венеры (кг/м³)

# Диапазоны значений
n_values = np.linspace(0, 200, 100)  # Частота вращения 0-50 Гц (0-3000 об/мин)
N_pot_values = np.arange(1, 9, 1)   # Количество винтов 1-8

def calculate_takeoff_mass(rho_sv, v_sigma, h, n, N_pot):
    """Расчёт взлётной массы"""
    return pi * rho_sv * (v_sigma**2 * h**2 * n**2 * N_pot) / g_B

# 1. Зависимость от частоты вращения
M_n = [calculate_takeoff_mass(rho_sv_venus, v_sigma_fixed, h_fixed, n, N_pot_fixed)
       for n in n_values]

# 2. Зависимость от количества винтов
M_N_pot = [calculate_takeoff_mass(rho_sv_venus, v_sigma_fixed, h_fixed, 81, int(N_pot))
           for N_pot in N_pot_values]

# Построение графиков
plt.figure(figsize=(14, 6))

# График 1: Частота вращения
plt.subplot(1, 2, 1)
plt.plot(n_values, M_n, 'b-', linewidth=2)
plt.xlabel('Частота вращения (Гц)', fontsize=12)
plt.ylabel('Взлётная масса (кг)', fontsize=12)
plt.title('Зависимость взлётной массы от частоты вращения', fontsize=12)
plt.grid(True)
plt.legend()

# График 2: Количество винтов
plt.subplot(1, 2, 2)
plt.bar(N_pot_values, M_N_pot, color='orange')
plt.xlabel('Количество винтов', fontsize=12)
plt.ylabel('Взлётная масса (кг)', fontsize=12)
plt.title('Зависимость взлётной массы от количества винтов', fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()