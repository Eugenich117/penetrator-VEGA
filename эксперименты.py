import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Фиксированные параметры (из статьи)
eta_en = 0.6       # КПД двигательной установки
U = 10             # Напряжение батареи, В
g = 8.871           # Ускорение свободного падения, м/с²
h = 0.1            # Шаг винта, м
Mkv = 0.7          # Масса конструкции без батареи, кг
k = 4              # Количество винтов
r = 0.1035         # Радиус винта, м
a, b = 69, 16.37   # Коэффициенты для массы батареи

# Диапазон ёмкостей батареи (1-30 Ач)
C_values = np.linspace(1, 300, 100)

# 1. Расчёт массы батареи (формула 13)
Mbat = (a * C_values + b) / 1000  # в кг

# 2. Расчёт полной массы аппарата
M = Mkv + Mbat

# 3. Расчёт частоты вращения винтов (формула 14)
n = 76 * np.sqrt((Mkv + Mbat) / Mkv)  # в об/с

# 4. Расчёт времени полёта (формула 12)
t_flight = (C_values * U * eta_en * 3600) / (M * g * h * n)

# Ограничения
M = np.clip(M, 0, 20)            # Макс. масса 20 кг
t_flight = np.clip(t_flight, 0, 10000)  # Макс. время 10000 сек

# График 1: Масса батареи от ёмкости
plt.figure(figsize=(10, 6))
plt.plot(C_values, Mbat, linewidth=2)
plt.xlabel('Ёмкость батареи, Ач', fontsize=12)
plt.ylabel('Масса батареи, кг', fontsize=12)
plt.title('Зависимость массы батареи от ёмкости', fontsize=14)
plt.grid(True)
plt.show()

# График 2: Полная масса от ёмкости
plt.figure(figsize=(10, 6))
plt.plot(C_values, M, linewidth=2)
plt.axhline(y=20, color='r', label='Макс. масса (20 кг)')
plt.xlabel('Ёмкость батареи, Ач', fontsize=12)
plt.ylabel('Полная масса, кг', fontsize=12)
plt.title('Зависимость полной массы от ёмкости', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# График 3: Частота вращения от ёмкости
plt.figure(figsize=(10, 6))
plt.plot(C_values, n*60, linewidth=2)  # Переводим в об/мин
plt.xlabel('Ёмкость батареи, Ач', fontsize=12)
plt.ylabel('Частота вращения, об/мин', fontsize=12)
plt.title('Зависимость частоты вращения от ёмкости', fontsize=14)
plt.grid(True)
plt.show()

# График 4: Время полёта от ёмкости
plt.figure(figsize=(10, 6))
plt.plot(C_values, t_flight/60, linewidth=2)  # Переводим в минуты
plt.axhline(y=10000/60, color='k', label='Макс. время (166.7 мин)')
plt.xlabel('Ёмкость батареи, Ач', fontsize=12)
plt.ylabel('Время полёта, мин', fontsize=12)
plt.title('Зависимость времени полёта от ёмкости', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 3D Визуализация: Время полёта от массы батареи и частоты вращения
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Создаем сетку для 3D графика
Mbat_3d = np.linspace(0.1, 5, 50)
n_3d = np.linspace(4000/60, 10000/60, 50)  # Переводим в об/с
Mbat_grid, n_grid = np.meshgrid(Mbat_3d, n_3d)
M_total = Mkv + Mbat_grid

# Рассчитываем время полёта для 3D графика (фиксируем C=20 Ач)
t_3d = (20 * U * eta_en * 3600) / (M_total * g * h * n_grid)
t_3d = np.clip(t_3d, 0, 10000)/60  # Переводим в минуты

# Построение поверхности
surf = ax.plot_surface(Mbat_grid, n_grid*60, t_3d, cmap='viridis', alpha=0.8)
ax.set_xlabel('Масса батареи, кг', fontsize=12)
ax.set_ylabel('Частота вращения, об/мин', fontsize=12)
ax.set_zlabel('Время полёта, мин', fontsize=12)
ax.set_title('Зависимость времени полёта от массы батареи и частоты вращения', fontsize=14)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Фиксированные параметры
h_fixed = 0.1      # Шаг винта (м)
n_fixed = 81       # Частота вращения (Гц)
N_pot_fixed = 4    # Количество винтов

# Диапазоны варьируемых параметров
rho_values = np.linspace(1, 30, 50)      # Плотность атмосферы (кг/м³)
r_values = np.linspace(0.05, 0.5, 50)     # Радиус винта (м)

# Создаем сетку для 3D графика
rho_grid, r_grid = np.meshgrid(rho_values, r_values)

# Функция расчёта взлётной массы
def calculate_takeoff_mass(rho, r):
    return m.pi * rho * (r**2 * h_fixed**2 * n_fixed**2 * N_pot_fixed) / g

# Рассчитываем массив взлётных масс
M_takeoff = calculate_takeoff_mass(rho_grid, r_grid)

# Создание фигуры
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности с разворотом на 180 градусов
surf = ax.plot_surface(r_grid, rho_grid, M_takeoff,
                      cmap='viridis',
                      alpha=0.8,
                      linewidth=0,
                      antialiased=True)

# Настройка осей (развернутых на 180°)
ax.set_xlabel('Радиус винта (м)', fontsize=12)
ax.set_ylabel('Плотность атмосферы (кг/м³)', fontsize=12)
ax.set_zlabel('Взлётная масса (кг)', fontsize=12)

# Угол обзора с поворотом на 180° (азимут = 135° вместо -45°)
ax.view_init(elev=25, azim=135)  # 180° разворот от предыдущего -45°

# Инвертирование оси X и Y для полного разворота
ax.invert_xaxis()
ax.invert_yaxis()

# Цветовая шкала
cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
cbar.set_label('Взлётная масса (кг)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()