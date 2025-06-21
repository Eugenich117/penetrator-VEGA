import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Параметры
temp_night_start = 330     # начальная температура, К
temp_night_min = 300       # минимальная температура ночью, К
temp_day_max = 360         # максимальная температура днем, К

g = 8.87                   # ускорение свободного падения, м/с^2
Rb = 6_051_800             # радиус Венеры, м
ro = 1.225                 # плотность атмосферы, кг/м^3
R = Rb + 50_000            # высота, м
U = 502                    # объем, м^3

# Временные параметры
total_seconds = 86400  # 24 часа
t_night1 = 6 * 3600    # 0:00 – 6:00
t_day = 12 * 3600      # 6:00 – 18:00
t_night2 = 6 * 3600    # 18:00 – 24:00

# Температура в зависимости от времени
def temperature(t):
    if t < t_night1:
        # Ночь: охлаждение от 330 до 300
        return temp_night_start - (temp_night_start - temp_night_min) * (t / t_night1)
    elif t < t_night1 + t_day:
        # День: нагрев от 300 до 360
        return temp_night_min + (temp_day_max - temp_night_min) * ((t - t_night1) / t_day)
    else:
        # Вечер: охлаждение от 360 до 300
        return temp_day_max - (temp_day_max - temp_night_min) * ((t - t_night1 - t_day) / t_night2)

# Подъемная сила
def lift_force(t):
    temp = temperature(t)
    g_eff = g * (Rb ** 2 / R ** 2)  # Эффективное ускорение
    rho_gas_factor = 176 / temp      # Упрощённая модель: плотность газа ∝ 1/T
    return ro * (1 - rho_gas_factor) * U * g_eff

# Адаптивный временной массив (исключаем дубликаты)
time1 = np.linspace(0, t_night1, 500)  # 0:00 – 6:00 (включая t_night1)
time2 = np.linspace(t_night1, t_night1 + t_day, 600)[1:]  # 6:00 – 18:00 (исключаем первую точку t_night1)
time3 = np.linspace(t_night1 + t_day, total_seconds, 500)[1:]  # 18:00 – 24:00 (исключаем первую точку t_night1 + t_day)

# Объединяем массивы
time = np.concatenate([time1, time2, time3])

# Убедимся, что time строго возрастающий
time = np.unique(time)  # Удаляем возможные дубликаты

# Вычисляем значения
temps_raw = np.array([temperature(t) for t in time])
force_raw = np.array([lift_force(t) for t in time])

# Проверка данных
print("Температура (min, max):", temps_raw.min(), temps_raw.max())
print("Подъемная сила (min, max):", force_raw.min(), force_raw.max())
print("Есть ли NaN в temps_raw?", np.any(np.isnan(temps_raw)))
print("Есть ли NaN в force_raw?", np.any(np.isnan(force_raw)))

# Интерполяция
interp_temp = PchipInterpolator(time, temps_raw)
interp_force = PchipInterpolator(time, force_raw)

# Более плотная сетка для сглаженных графиков
time_smooth = np.linspace(0, total_seconds, 2000)
temps_smooth = interp_temp(time_smooth)
force_smooth = interp_force(time_smooth)

# Графики
plt.figure(figsize=(6, 4))

# Подъемная сила
plt.subplot(2, 1, 1)
plt.plot(time_smooth / 3600, force_smooth, label='Подъемная сила (Н)')
plt.ylabel('Сила (Н)')
plt.title('Подъемная сила дирижабля в течение суток')
plt.grid(True)
plt.legend()

# Температура
plt.subplot(2, 1, 2)
plt.plot(time_smooth / 3600, temps_smooth, color='orange', label='Температура (K)')
plt.xlabel('Время (часы)')
plt.ylabel('Температура (K)')
plt.title('Температура газа внутри оболочки')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()