import numpy as np
import matplotlib.pyplot as plt

# Константы
g_B = 8.871  # Ускорение свободного падения, м/с²
pi = np.pi  # Математическая константа

# Базовые значения для фиксированных параметров
v_sigma_base = 40.0  # Базовая суммарная скорость (м/с)
h_base = 500.0      # Базовая высота (м)
n_base = 0.5        # Базовый параметр n
N_pot_base = 5.0    # Базовая потеря N_ПОТ

# Диапазоны значений
rho_sv_values = np.arange(0.5, 80.1, 1.0)  # Плотность среды (кг/м³): 0.5, 1.5, ..., 80
v_sigma_values = np.arange(0.05, 0.5, 0.01)      # радиус винта м
h_values = np.arange(2100, 10000, 10)         # шаг винта метры
n_values = np.arange(0, 1, 0.1)            # частота вращения
N_pot_values = np.arange(0, 10, 1)         # количество винтов

# Функция для расчёта взлётной массы
def calculate_takeoff_mass(rho_sv, v_sigma, h, n, N_pot):
    numerator = v_sigma**2 * h**2 * n**2 * N_pot
    if numerator <= 0:
        return 0.0  # Избегаем отрицательных или неопределённых значений
    return pi * rho_sv * (numerator / g_B)

# Расчёт взлётной массы для каждого параметра
M_rho = [calculate_takeoff_mass(rho, v_sigma_base, h_base, n_base, N_pot_base) for rho in rho_sv_values]
M_v = [calculate_takeoff_mass(rho_sv_values[0], v, h_base, n_base, N_pot_base) for v in v_sigma_values]
M_h = [calculate_takeoff_mass(rho_sv_values[0], v_sigma_base, h, n_base, N_pot_base) for h in h_values]
M_n = [calculate_takeoff_mass(rho_sv_values[0], v_sigma_base, h_base, n, N_pot_base) for n in n_values]
M_N_pot = [calculate_takeoff_mass(rho_sv_values[0], v_sigma_base, h_base, n_base, N_pot) for N_pot in N_pot_values]

# График 1: Зависимость от плотности среды
plt.figure(figsize=(10, 6))
plt.plot(rho_sv_values, M_rho, label='Зависимость взлетной массы от плотности среды (ρ_СВ)', color='blue', marker='o')
plt.xlabel('Плотность среды (ρ_СВ, кг/м³)')
plt.ylabel('Взлётная масса (кг)')
plt.title(r'Зависимость взлётной массы от плотности среды $\frac{кг}{м^3}$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График 2: Зависимость от суммарной скорости
plt.figure(figsize=(10, 6))
plt.plot(v_sigma_values, M_v, label='Зависимость взлетной массы от радиуса винта', color='green', marker='s')
plt.xlabel('Радиус винта (м)')
plt.ylabel('Взлётная масса (кг)')
plt.title('Зависимость взлётной массы от радиуса винта')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График 3: Зависимость от высоты
plt.figure(figsize=(10, 6))
plt.plot(h_values, M_h, label='Зависимость взлетной массы от шага винта (h)', color='red', marker='^')
plt.xlabel('Шаг винта (h, м)')
plt.ylabel('Взлётная масса (кг)')
plt.title('Зависимость взлётной массы от шага винта')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График 4: Зависимость от параметра n
plt.figure(figsize=(10, 6))
plt.plot(n_values, M_n, label='Зависимость взлетной массы от частоты вращения ротора', color='purple', marker='d')
plt.xlabel('Частота вращения ротора (Гц)')
plt.ylabel('Взлётная масса (кг)')
plt.title('Зависимость взлётной массы от параметра n')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График 5: Зависимость от количества винтов
plt.figure(figsize=(10, 6))
plt.plot(N_pot_values, M_N_pot, label='Зависимость взлетной массы от количества винтов (N_ПОТ)', color='orange', marker='x')
plt.xlabel('Количество винтов')
plt.ylabel('Взлётная масса (кг)')
plt.title('Зависимость взлётной массы от количества винтов')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()