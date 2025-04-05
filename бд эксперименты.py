import numpy as np
import time
from scipy.special import sindg, cosdg
import math as m
import matplotlib.pyplot as plt
from icecream import ic
import scipy
import concurrent.futures
import threading
import multiprocessing
from multiprocessing import Queue, Pipe
import random
import bisect
import sqlite3
import pandas as pd
from scipy.integrate import quad
import gc
import sqlite3
# мат модель из книжки воронцова упрощенная



# Cxa = 1.3#((2*L*r2*(1+r1/r2)/S))*(m.tan(Qk)/2)*(2*m.cos(0)**2*m.sin(Qk)**2+m.sin(0))
# Cya = 0#((2*L*r2*(1+r1/r2))/S)*m.pi*m.cos(0)*m.sin(0)*m.cos(Qk)*m.cos(Qk)
# Px = mass / Cxa * S
# K = Cya / Cxa
# ic(Cxa, Cya)



def find_closest_points_ro(x, xi):
    """
    Находит 4 ближайшие точки к xi в списке x.
    """
    closest_points = []
    for i in range(len(x)):
        if xi - 2 <= x[i] <= xi + 2:
            if i <= 2:
                closest_indices = list(range(4))
            elif i >= len(x) - 2:
                closest_indices = list(range(len(x) - 4, len(x)))
            else:
                closest_indices = list(range(i - 2, i + 2))
            closest_points = [x[idx] if 0 <= idx < len(x) else closest_points[-1] for idx in closest_indices]
            break
    return closest_points


def divided_diff_ro(x, y):
    """
    Вычисление разделённых разностей.
    """
    n = len(y)
    coef = [0] * n
    coef[0] = y[0]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            if x[i] == x[i - j]:
                coef[i] = y[i]  # Просто присвоить значение y[i], чтобы избежать деления на ноль
            else:
                y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])
                coef[i] = y[i]
    return coef


def newton_interpolation_ro(x, y, xi):
    """
    Интерполяция методом Ньютона.
    """
    closest_points = find_closest_points_ro(x, xi)
    x_interpolate = closest_points
    y_interpolate = [y[x.index(x_interpolate[0])], y[x.index(x_interpolate[1])], y[x.index(x_interpolate[2])],
                     y[x.index(x_interpolate[3])]]
    coef = divided_diff_ro(x_interpolate, y_interpolate)
    n = len(coef) - 1
    result = coef[n]
    for i in range(n - 1, -1, -1):
        result = result * (xi - x_interpolate[i]) + coef[i]
    return result


def find_closest_points(x, xi):
    """
    Находит две ближайшие точки к xi в списке x.
    """
    closest_points = []
    for i in range(len(x)):
        if x[i] >= xi:
            if i == 0:
                closest_points = [x[i], x[i + 1]]
            else:
                closest_points = [x[i - 1], x[i]]
            break
    return closest_points


def divided_diff(x, y):
    """
    Вычисление разделённых разностей.
    """
    n = len(y)
    coef = [0] * n
    coef[0] = y[0]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            if x[i] == x[i - j]:
                coef[i] = y[i]  # Просто присвоить значение y[i], чтобы избежать деления на ноль
            else:
                y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])
                coef[i] = y[i]
    return coef


def newton_interpolation(x, y, xi):
    """
    Интерполяция методом Ньютона.
    """
    closest_points = find_closest_points(x, xi)
    x_interpolate = closest_points
    y_interpolate = [y[x.index(x_interpolate[0])], y[x.index(x_interpolate[1])]]
    coef = divided_diff(x_interpolate, y_interpolate)
    n = len(coef) - 1
    result = coef[n]

    for i in range(n - 1, -1, -1):
        result = result * (xi - x_interpolate[i]) + coef[i]

    return result


def Get_ro(R): # В основной функции всё в метрах, в полиноме в километрах
    x = [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86,
         84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
         31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
         2, 1, 0]
    y = [3.97200000e-08, 7.890 * 10 ** (-7), 1.35931821e-06, 1.77164551e-06, 2.30904567e-06, 3.00945751e-06, 3.92232801e-05,
         5.11210308e-05, 6.66277727e-05, 8.68382352e-05, 1.13179216e-04, 1.47510310e-04, 1.92255188e-04, 2.50572705e-04,
         3.26579903e-04, 1.347 * 10 ** (-4), 0.0002, 0.0004, 0.0007, 0.0012, 0.0019, 0.0031, 0.0049, 0.0077,
         0.0119, 0.0178, 0.0266, 0.0393, 0.0578, 0.0839, 0.1210, 0.1729, 0.2443, 0.3411, 0.4694, 0.6289,
         0.8183, 1.0320, 1.2840, 1.5940, 1.9670, 2.4260, 2.9850, 3.6460, 4.4040, 5.2760, 6.2740, 7.4200,
         8.7040, 9.4060, 10.1500, 10.9300, 11.7700, 12.6500, 13.5900, 14.5700, 15.6200, 16.7100, 17.8800,
         19.1100, 20.3900, 21.7400, 23.1800, 24.6800, 26.2700, 27.9500, 29.7400, 31.6000, 33.5400,
         35.5800, 37.7200, 39.9500, 42.2600, 44.7100, 47.2400, 49.8700, 52.6200, 55.4700, 58.4500, 61.5600,
         64.7900]
    ro = newton_interpolation_ro(x, y, R / 1000)
    return ro


def Cx(xi, V_sound):
    x = [0, 0, 0.2, 0.4, 0.6, 0.1, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    y = [0.75, 0.8, 0.9, 1.1, 1.3, 1.45, 1.52, 1.55, 1.6, 1.7, 1.8, 1.78, 1.75, 1.7, 1.65, 1.6, 1.55, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52]
    return newton_interpolation(x, y, xi/V_sound)


def Cx_wind(xi, V_sound):
    x = [0, 0, 0.2, 0.4, 0.6, 0.1, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    y = [0.15, 0.15, 0.18, 0.3, 0.38, 0.81, 0.92, 0.97, 0.995, 0.991, 0.985, 0.98, 0.975, 0.97, 0.955, 0.935, 0.925,
         0.91, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9]
    return newton_interpolation(x, y, xi/V_sound)


def pressure_func(xi):
    x = [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86,
         84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
         31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
         2, 1, 0]
    y = [0.0019907, 0.005, 0.10, 0.30, 0.50, 0.70, 0.90, 1.10, 1.30, 1.50, 1.70, 1.90, 2.10, 2.30, 2.50, 2.66, 4.45,
         7.519, 12.81, 20.0, 40.0, 60.0, 110.0, 170.0, 280.0, 450.0, 700.0, 1080.0, 1650.0, 2480.0, 3690.0, 5450.0,
         7970.0, 11560.0, 16590.0, 23570.0, 33060.0, 45590.0, 61600.0, 81670.0, 106600.0, 137500.0, 175600.0, 222600.0,
         280200.0, 350100.0, 434200.0, 534600.0, 653700.0, 794000.0, 872900.0, 958100.0, 1050000.0, 1149000.0,
         1256000.0, 1370000.0, 1493000.0, 1625000.0, 1766000.0, 1917000.0, 2079000.0, 2252000.0, 2436000.0,
         2633000.0, 2843000.0, 3066000.0, 3304000.0, 3557000.0, 3826000.0, 4112000.0, 4416000.0, 4739000.0,
         5081000.0, 5444000.0, 5828000.0, 6235000.0, 6665000.0, 7120000.0, 7601000.0, 8109000.0, 8645000.0, 9210000.0]
    return newton_interpolation_ro(x, y, xi / 1000)


def v_sound(R):
    x = [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86,
         84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
         31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
         2, 1, 0]
    y = [174, 176, 178, 180, 182, 185, 186, 187, 190, 193, 195, 196, 198, 199, 201, 203, 205, 206, 208.0, 208.0, 209.0,
         212.2, 215.4, 218.6, 221.8, 225.0, 228.2, 231.4, 234.6, 237.8, 241.0, 244.0, 247.0, 250.0, 253.0, 256.0, 263.2,
         270.4, 277.6, 284.8, 292.0, 296.8, 301.6, 306.4, 311.2, 316.0, 321.2, 326.4, 331.6, 336.8, 339.4, 342.0, 344.6,
         347.2, 349.8, 352.4, 355.0, 357.4, 359.8, 362.2, 364.6, 367.0, 369.4, 371.8, 374.2, 376.6, 379.0, 381.0, 383.0,
         385.0, 387.0, 389.0, 391.2, 393.4, 395.6, 397.8, 400.0, 402.0, 404.0, 406.0, 408.0, 410.0]
    return newton_interpolation_ro(x, y, R / 1000)


start_time = time.time()
r1 = 0.4
mass = 120
h = 125_000
mass_planet = 4.867 * 10 ** 24
Rb = 6_051_800
gravy_const = 6.67 * 10 ** (-11)
g = 8.869
dt = 0.01
tetta = -9
x = 0
y = 0
plotnost = []; CX = []

cToDeg = 180 / m.pi
cToRad = m.pi / 180

R = Rb + h
dV = 0


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def dV_func(initial):
    S = initial['S']
    R = initial['R']
    Cxa = initial['Cxa']
    ro = initial['ro']
    V = initial['V']
    tetta = initial['tetta']
    mass = initial['mass']
    V_wind = initial['V_wind']
    wind_angle = initial['wind_angle']
    Cxa_wind = initial['Cxa_wind']
    V_wind_x = V_wind * m.sin(wind_angle)#Вдольтраектории
    P = initial['P']
    #dV=((-1/(2*Px))*Cxa*ro*V**2-((gravy_const*mass_planet)/R**2)*scipy.special.sindg(tetta))*dt#ОСНОВНАЯМОДЕЛЬКОСЕНКОВОЙ
    dV = (P - mass * (g * Rb ** 2 / R ** 2) * m.sin(tetta) - (0.5 * ro * V ** 2 * Cxa * S) + sign(V_wind_x) * (
                0.5 * ro * V_wind_x ** 2 * Cxa_wind * S)) / mass
    return dV, 'V'


def dL_func(initial):
    V=initial['V']
    tetta=initial['tetta']
    V_wind=initial['V_wind']
    wind_angle=initial['wind_angle']

    V_wind_z=V_wind*m.cos(wind_angle)#Перпендикулярнотраектории
    dL=m.sqrt(V**2+V_wind_z**2)*Rb/R*m.cos(tetta)
    return dL, 'L'


def dtetta_func(initial):
    V=initial['V']
    tetta=initial['tetta']
    R=initial['R']
    V_wind=initial['V_wind']

    dtetta=((-(g*Rb**2/R**2)*m.cos(tetta))/m.sqrt(V**2+V_wind**2)+(m.sqrt(V**2+V_wind**2)/R))
    #dtetta=(((V**2-((gravy_const*mass_planet)/R**2)*R)/(V*R))*scipy.special.cosdg(tetta))*dt
    return dtetta, 'tetta'


def dR_func(initial):
    V=initial['V']
    tetta=initial['tetta']
    V_wind=initial['V_wind']
    wind_angle=initial['wind_angle']

    V_wind=V_wind*m.cos(wind_angle)#Вдольтраектории
    dR=(m.sqrt(V**2+V_wind**2)*m.sin(tetta))
    return dR, 'R'


def qk_func(initial):
    ro = initial['ro']
    V = initial['V']
    dqk_compression = (7.845 * 0.5) * (ro / 64.79) * (V / 1000) ** 8 # нагрев при ударной волне при входе с около параболических скоростей
    #dqk = ((1.318 * 10 ** 5) / m.sqrt(0.5)) * m.sqrt(ro / 64.79) * (V / 7328) ** 3.25  # для ламинарного обтекания
    dqk_turbulent = (1.15 * 10 ** 6) * ((ro ** 0.8)/(0.5 ** 0.2)) * (V / 7328) ** 3.19 #для турбулентного обтекания при входе с около параболических скоростей при нулевом угле атаки
    dqk = dqk_turbulent + dqk_compression
    return dqk, 'qk'


def quantity_func(qk):
    return qk


def wind(h, t, next_update_time, V_wind, wind_angle):
    bounds = [0, 2_000, 6_000, 10_000, 18_000, 28_000, 36_000, 42_000, 48_000,
              55_000, 61_000, 68_000, 76_000, 85_000, 94_000, 100_000, float('inf')]

    #Функциидлявычисленияv_windвзависимостиотдиапазона
    actions=[
        lambda: random.uniform(0, 3),  # 0 - 2 км
        lambda: random.uniform(3, 7),  # 2 - 6 км
        lambda: random.uniform(7, 15),  # 6 - 10 км
        lambda: random.uniform(15, 25),  # 10 - 18 км
        lambda: random.uniform(25, 35),  # 18 - 28 км
        lambda: random.uniform(30, 40),  # 28 - 36 км
        lambda: random.uniform(35, 50),  # 36 - 42 км
        lambda: random.uniform(45, 60),  # 42 - 48 км
        lambda: random.uniform(55, 70),  # 48 - 55 км
        lambda: random.uniform(65, 80),  # 55 - 61 км
        lambda: random.uniform(70, 90),  # 61 - 68 км
        lambda: random.uniform(85, 100),  # 68 - 76 км
        lambda: random.uniform(75, 85),  # 76 - 85 км
        lambda: random.uniform(60, 75),  # 85 - 94 км
        lambda: random.uniform(10, 20),  # 94 - 100 км
        lambda: 0  # Выше 100 км
    ]
    #Находим индекс диапазона с помощью bisect
    index = bisect.bisect_right(bounds, h)-1

    if t >= next_update_time:
        V_wind = actions[index]()  # Генерируем новую скорость ветра
        wind_angle = random.uniform(0, 2 * m.pi)  # Генерируем новый угол ветра
        wind_timer = random.uniform(0.2, 10)  # Случайный таймер
        next_update_time = t + wind_timer  # Устанавливаем время следующего обновления

    if h >= 50_000:
        wind_angle = random.uniform((-m.pi / 6), (m.pi / 6))

    return V_wind, wind_angle, next_update_time


def runge_kutta_4(equations, initial, dt, dx):
    '''equations - это список названий функций с уравнениями для системы
    initial это переменные с начальными условиями
    dx - это список переменных, которые будут использованы для интегрирования уравнения'''
    k1 = {key: 0 for key in initial.keys()}
    k2 = {key: 0 for key in initial.keys()}
    k3 = {key: 0 for key in initial.keys()}
    k4 = {key: 0 for key in initial.keys()}

    derivatives_1 = {key: initial[key] for key in initial}
    derivatives_2 = {key: initial[key] for key in initial}
    derivatives_3 = {key: initial[key] for key in initial}
    derivatives_4 = {key: initial[key] for key in initial}

    new_values = [0] * len(equations)

    for i, eq in enumerate(equations):
        derivative, key = eq(initial)
        k1[key] += derivative
        derivatives_1[key] = initial[key] + derivative * dt / 2
        derivatives_1[dx[i]] += dt / 2
        # derivatives_1 = {key: value / 2 for key, value in derivatives_1.items()}

    for i, eq in enumerate(equations):
        derivative, key = eq(derivatives_1)
        k2[key] += derivative
        derivatives_2[key] = initial[key] + derivative * dt / 2
        derivatives_2[dx[i]] += dt / 2
        # derivatives_2 = {key: value / 2 for key, value in derivatives_2.items()}

    for i, eq in enumerate(equations):
        derivative, key = eq(derivatives_2)
        k3[key] += derivative
        derivatives_3[key] = initial[key] + derivative * dt
        derivatives_3[dx[i]] += dt

    for i, eq in enumerate(equations):
        derivative, key = eq(derivatives_3)
        k4[key] += derivative
        derivatives_4[key] = initial[key] + derivative * dt
        new_values[i] = initial[key] + (1 / 6) * dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
    return new_values


def save_results_to_db(results):
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()

    # Создание таблицы, если её ещё нет
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            best_V REAL,
            best_P REAL,
            p_soplar REAL,
            tetta REAL,
            L REAL,
            H REAL,
            t REAL,
            mass REAL,
            chromosome TEXT,
            population_size INTEGER,
            generations INTEGER,
            mutation_rate REAL,
            elapsed_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PARAM_BOUNDS TEXT
        )
    ''')

    # Вставка данных
    cursor.execute('''
        INSERT INTO optimization_results (
            best_V, best_P, p_soplar, tetta, L, H, t, mass,
            chromosome, population_size, generations, mutation_rate, elapsed_time, PARAM_BOUNDS
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', results)

    conn.commit()
    conn.close()


#(i, napor, TETTA, X, Y, V_MOD, T, PX, nx, acceleration) # передача листов для результатов в явном виде в функцию
def compute_trajectory(equations, dx, pipe_conn, chromosome):
    #print(f"поток {i} запущен")
    t = 0
    mass = 150  + (chromosome['mass_consumption'] * 30)
    mass_stop = 150 + ((chromosome['mass_consumption'] * 30)/3)
    d = 0.8
    S = (m.pi * d ** 2) / 4
    I_ud, P, V, tetta, R, L = 230, 0, random.uniform(10_900, 11_100), random.uniform(-25, -15) * cToRad, Rb + h, 0
    print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}')
    initial = {}
    initial['S'] = S
    initial['mass'] = mass

    local_TETTA = []; local_X = []; local_Y = []; local_V_MOD = []; local_T = []; local_napor = []; local_nx = []
    local_PX = []; local_acceleration = []; local_v_wind = []; local_wind_angle = []; local_Quantitiy_warm = []
    local_Tomega = []; local_Qk = []; local_P = []
    lam, phi, epsilon = 0, 0, 0
    omega_b = 2.9926 * 10 ** -7  # Угловая скорость вращения планеты, рад/с
    next_update_time = -1
    V_wind = 0
    wind_angle = 0
    v_gas = 0
    p_soplar = 0
    mass_consumption = 0
    S_soplar = 0
    qk = 0
    h_vcl = chromosome['h_vcl']
    h_stop = chromosome['h_stop']
    while R >= Rb + h_vcl:
        pressure = pressure_func(R - Rb)
        V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
        V_sound = v_sound(R - Rb)
        ro = Get_ro(R - Rb)
        Cxa = Cx(V, V_sound)
        Cxa_wind = Cx_wind(V, V_sound)
        Px = mass / Cxa * S
        initial.update({'Px': Px, 'I_ud': I_ud, 'P': P, 'lam': lam, 'phi': phi, 'epsilon': epsilon, 'V_wind': V_wind,
            'omega_b': omega_b, 'wind_angle': wind_angle, 'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro,
            'mass': mass, 'L': L, 'V': V, 'R': R, 'pressure': pressure, 'v_gas': v_gas, 'p_soplar': p_soplar, 'qk': qk,
            'mass_consumption': mass_consumption, 'S_soplar': S_soplar})
        values = runge_kutta_4(equations, initial, dt, dx)
        V = values[0]
        L = values[1]
        tetta = values[2]
        R = values[3]
        qk = values[4]
        P = v_gas * mass_consumption + S_soplar * (p_soplar - pressure) #(P / ((g * Rb**2/R**2) * I_ud))
        t += dt

        local_P.append(P)
        quantity_warm, error = quad(quantity_func, 0, t)
        local_Quantitiy_warm.append(quantity_warm)
        local_Tomega.append((qk / (0.8 * 5.67 * 10 ** (-8))) ** 0.25)
        local_Qk.append(qk)
        local_wind_angle.append(wind_angle)
        local_v_wind.append(V_wind)
        local_TETTA.append(tetta * cToDeg)
        local_X.append(L)
        local_Y.append(R - Rb)
        local_V_MOD.append(V)
        local_T.append(t)
        local_napor.append(0.5 * ro * V ** 2)
        local_nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
        local_PX.append(Px)


    print(f' без движка V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}, mass = {mass:.3f}')
    mass_consumption = chromosome['mass_consumption']
    p_soplar = pressure_func(h_vcl - ((h_vcl - h_stop)/2))
    S_soplar = (mass_consumption * 1550 * 1.8) / 2.533e+7
    while R >= Rb + h_stop:
        pressure = pressure_func(R - Rb)
        V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
        V_sound = v_sound(R - Rb)
        ro = Get_ro(R - Rb)
        Cxa = Cx(V, V_sound)
        Cxa_wind = Cx_wind(V, V_sound)
        Px = mass / Cxa * S
        v_gas = m.sqrt(2 * 2000 * 3500 * (1 - (pressure / 2.533e+7) ** (0.2 / 1.2)))
        initial.update({'Px': Px, 'I_ud': I_ud, 'P': P, 'lam': lam, 'phi': phi, 'epsilon': epsilon, 'V_wind': V_wind,
            'omega_b': omega_b, 'wind_angle': wind_angle, 'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro,
            'mass': mass, 'L': L, 'V': V, 'R': R, 'pressure': pressure, 'v_gas': v_gas, 'p_soplar': p_soplar, 'qk': qk,
            'mass_consumption': mass_consumption, 'S_soplar': S_soplar})
        values = runge_kutta_4(equations, initial, dt, dx)
        V = values[0]
        L = values[1]
        tetta = values[2]
        R = values[3]
        qk = values[4]
        mass -= mass_consumption * dt
        P = v_gas * mass_consumption + S_soplar * (p_soplar - pressure)  # (P / ((g * Rb**2/R**2) * I_ud))
        t += dt
        #print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}, mass = {mass:.3f}')
        local_P.append(P)
        quantity_warm, error = quad(quantity_func, 0, t)
        local_Quantitiy_warm.append(quantity_warm)
        local_Tomega.append((qk / (0.8 * 5.67 * 10 ** (-8))) ** 0.25)
        local_Qk.append(qk)
        local_wind_angle.append(wind_angle)
        local_v_wind.append(V_wind)
        local_TETTA.append(tetta * cToDeg)
        local_X.append(L)
        local_Y.append(R - Rb)
        local_V_MOD.append(V)
        local_T.append(t)
        local_napor.append(0.5 * ro * V ** 2)
        local_nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
        local_PX.append(Px)
        if (mass <= (mass_stop)) or (R <= Rb):
            h_stop = (R - Rb)
            break

    print(f' с движком V = {V:.3f}, P = {np.max(local_P)}, давление на срезе сопла = {p_soplar:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}, mass = {mass:.3f}')

    v_gas = 0
    p_soplar = 0
    mass_consumption = 0
    S_soplar = 0
    while R >= Rb:
        pressure = pressure_func(R - Rb)
        V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
        V_sound = v_sound(R - Rb)
        ro = Get_ro(R - Rb)
        Cxa = Cx(V, V_sound)
        Cxa_wind = Cx_wind(V, V_sound)
        Px = mass / Cxa * S
        initial.update({'Px': Px, 'I_ud': I_ud, 'P': P, 'lam': lam, 'phi': phi, 'epsilon': epsilon, 'V_wind': V_wind,
            'omega_b': omega_b, 'wind_angle': wind_angle, 'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro,
            'mass': mass, 'L': L, 'V': V, 'R': R, 'pressure': pressure, 'v_gas': v_gas, 'p_soplar': p_soplar,  'qk': qk,
            'mass_consumption': mass_consumption, 'S_soplar': S_soplar})
        values = runge_kutta_4(equations, initial, dt, dx)
        V = values[0]
        L = values[1]
        tetta = values[2]
        R = values[3]
        qk = values[4]
        mass -= mass_consumption
        P = v_gas * mass_consumption + S_soplar * (p_soplar - pressure) #(P / ((g * Rb**2/R**2) * I_ud))
        t += dt
        #print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}, mass = {mass:.3f}')

        quantity_warm, error = quad(quantity_func, 0, t)
        local_Quantitiy_warm.append(quantity_warm)
        local_Tomega.append((qk / (0.8 * 5.67 * 10 ** (-8))) ** 0.25)
        local_Qk.append(qk)
        local_wind_angle.append(wind_angle)
        local_v_wind.append(V_wind)
        local_TETTA.append(tetta * cToDeg)
        local_X.append(L)
        local_Y.append(R - Rb)
        local_V_MOD.append(V)
        local_T.append(t)
        local_napor.append(0.5 * ro * V ** 2)
        local_nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
        local_PX.append(Px)

    print(f' без движка V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}, mass = {mass:.3f}')

    for j in range(1, len(local_V_MOD)):
        derivative_value = (local_V_MOD[j] - local_V_MOD[j - 1]) / dt
        local_acceleration.append(derivative_value)

    result = (
        local_V_MOD[-1],  # Последнее значение скорости V
        local_P[-1],      # Последнее значение давления P
        p_soplar,         # Давление на срезе сопла
        tetta * cToDeg,   # Угол тетта в градусах
        L,                # Дальность
        (R - Rb),         # Высота
        t,                # Время
        (mass + (chromosome['mass_consumption'] * 30))              # Масса
    )
    try:
        # 1. Отправка данных + автоматическое закрытие трубы (лучшая практика)
        with pipe_conn:
            pipe_conn.send(result)

        # 2. Принудительная очистка памяти (опционально, для больших данных)
        del result  # Удаляем ссылку на кортеж
        gc.collect()  # Ускоряем освобождение памяти (если данные огромные)

    except (BrokenPipeError, EOFError) as e:
        # Обработка ошибок (например, если родительский процесс завершился раньше)
        if not pipe_conn.closed:
            print(f"Ошибка передачи данных в процессе {i}: {e}")
            result = (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
                      local_acceleration, local_v_wind, local_wind_angle, local_Quantitiy_warm, local_Tomega, local_Qk,
                      local_P)
            pipe_conn.send(result)  # Передаем данные
            pipe_conn.close()  # Закрываем трубу
    #queue.put(result, block=False)
    if not pipe_conn.closed:
        print("выполнен аварийный вариант")
        result = (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
                  local_acceleration, local_v_wind, local_wind_angle, local_Quantitiy_warm, local_Tomega, local_Qk,
                  local_P)
        pipe_conn.send(result)  # Передаем данные
        pipe_conn.close()  # Закрываем трубу

# Границы параметров
PARAM_BOUNDS = {
    'h_vcl': (500, 3000),
    'h_stop': (0, 50),
    'mass_consumption': (2.5, 7.5),
}

# Функция для создания случайной хромосомы
def create_chromosome():
    while True:
        h_vcl = random.uniform(*PARAM_BOUNDS['h_vcl'])
        h_stop_min = max(PARAM_BOUNDS['h_stop'][0], h_vcl - 15000)
        h_stop_max = min(PARAM_BOUNDS['h_stop'][1], h_vcl - 500)
        if h_stop_min <= h_stop_max:
            h_stop = random.uniform(h_stop_min, h_stop_max)
            return {
                'h_vcl': h_vcl,
                'h_stop': h_stop,
                'mass_consumption': random.uniform(*PARAM_BOUNDS['mass_consumption']),
            }


# Функция для создания начальной популяции
def create_population(size):
    return [create_chromosome() for _ in range(size)]


# Селекция (турнирный отбор)
def selection(population, fitness_scores):
    selected = []
    for _ in range(POPULATION_SIZE):
        idx1, idx2 = random.sample(range(POPULATION_SIZE), 2)
        if fitness_scores[idx1] > fitness_scores[idx2]:
            selected.append(population[idx1])
        else:
            selected.append(population[idx2])
    return selected


# Скрещивание (одноточечное)
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)  # Случайная точка кроссовера
    child1 = {}
    child2 = {}
    keys = list(parent1.keys())
    for i, key in enumerate(keys):
        if i < point:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]

    # Проверяем условия для h_vcl и h_stop
    if child1['h_vcl'] <= child1['h_stop'] or not (500 < (child1['h_vcl'] - child1['h_stop']) < 15000):
        # Если условия нарушены, повторяем кроссовер
        return crossover(parent1, parent2)
    if child2['h_vcl'] <= child2['h_stop'] or not (500 < (child2['h_vcl'] - child2['h_stop']) < 15000):
        return crossover(parent1, parent2)

    return child1, child2


def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    selected = random.choices(population, weights=probabilities, k=POPULATION_SIZE)
    return selected


# Мутация
def mutate(chromosome):
    for param in chromosome:
        if random.random() < MUTATION_RATE:
            if param == 'h_vcl' or param == 'h_stop':
                # Генерация нового значения с учетом ограничений
                while True:
                    new_value = random.uniform(*PARAM_BOUNDS[param])
                    if param == 'h_vcl':
                        h_stop = chromosome['h_stop']
                        if new_value > h_stop and 500 < (new_value - h_stop) < 15000:
                            chromosome[param] = new_value
                            break
                    elif param == 'h_stop':
                        h_vcl = chromosome['h_vcl']
                        if h_vcl > new_value and 500 < (h_vcl - new_value) < 15000:
                            chromosome[param] = new_value
                            break
            else:
                chromosome[param] = random.uniform(*PARAM_BOUNDS[param])
    return chromosome


# Основной цикл генетического алгоритма
def create_population(size):
    return [create_chromosome() for _ in range(size)]

# Функция приспособленности (запуск модели и получение V)
def fitness(chromosome, pipe_conn):
    # Здесь вызывается ваша функция compute_trajectory с параметрами из хромосомы
    # Например, можно передать параметры через initial или напрямую в compute_trajectory

    dx = ['V', 'L', 'tetta', 'R', 'qk']
    equations = [dV_func, dL_func, dtetta_func, dR_func, qk_func]
    result = compute_trajectory(equations, dx, pipe_conn, chromosome)
    pipe_conn.send(result)
    if result is not None and child_conn.writable:  # Проверяем, можно ли отправить данные
        child_conn.send(result)
    '''except Exception as e:
        print(f"Ошибка в fitness: {e}")'''

    '''finally:
        child_conn.close()  # Закрываем соединение'''

# Основной цикл генетического алгоритма
def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}")

        # Оценка приспособленности
        fitness_scores = []
        parent_conns = []
        child_conns = []
        last_V_values = []
        last_P_values = []
        p_soplar_values = []
        tetta_values = []
        L_values = []
        H_values = []
        t_values = []
        mass_values = []
        tasks = []

        # Создаем пул процессов
        pool = multiprocessing.Pool(processes=POPULATION_SIZE)

        for i, chromosome in enumerate(population):
            parent_conn, child_conn = multiprocessing.Pipe()
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)
            tasks.append((chromosome, child_conn))

        # Запускаем задачи асинхронно
        for task in tasks:
            pool.apply_async(fitness, task)

        # Получаем результаты
        for i, conn in enumerate(parent_conns):
            result = parent_conns[i].recv()
            # тут поменял result = conn.recv()
            # Распаковываем результат
            (last_V, last_P, p_soplar, tetta, L, H, t, mass) = result
            last_V_values.append(last_V)
            last_P_values.append(last_P)
            p_soplar_values.append(p_soplar)
            tetta_values.append(tetta)
            L_values.append(L)
            H_values.append(H)
            t_values.append(t)
            mass_values.append(mass)
            fitness_scores.append(last_V)
            '''print(f"отладка V: {last_V_values}")
            print(f"отладка P: {len(last_P_values), last_P_values}")
            print(f"отладка L: {len(L_values), L_values}")
            print(f"отладка H: {len(H_values), H_values}")
            print(f"отладка: {len(mass_values), mass_values}")'''

        # Селекция
        selected = selection(population, fitness_scores)

        # Скрещивание и мутация
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    # Возвращаем лучшую хромосому и параметры
    best_index = np.argmax(fitness_scores)
    best_chromosome = population[best_index]
    best_last_V = last_V_values[best_index]
    best_last_P = last_P_values[best_index]
    best_p_soplar = p_soplar_values[best_index]
    best_tetta = tetta_values[best_index]
    best_L = L_values[best_index]
    best_H = H_values[best_index]
    best_t = t_values[best_index]
    best_mass = mass_values[best_index]

    return (
        best_chromosome,
        best_last_V,
        best_last_P,
        best_p_soplar,
        best_tetta,
        best_L,
        best_H,
        best_t,
        best_mass
    )

def view_all_results():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM optimization_results')
    rows = cursor.fetchall()

    # Заголовки столбцов:
    column_names = [
        'id', 'best_last_V', 'best_last_P', 'best_p_soplar', 'best_tetta',
        'best_L', 'best_H', 'best_t', 'best_mass', 'best_chromosome',
        'population_size', 'generations', 'mutation_rate', 'elapsed_time', 'timestamp', 'PARAM_BOUNDS'
    ]

    for row in rows:
        print("=== Результат ===")
        for col_name, value in zip(column_names, row):
            print(f"{col_name}: {value}")
        print()

    conn.close()

# Запуск генетического алгоритма
if __name__ == '__main__':
    POPULATION_SIZE = 30
    GENERATIONS = 10
    MUTATION_RATE = 0.8

    (
        best_chromosome,
        best_last_V,
        best_last_P,
        best_p_soplar,
        best_tetta,
        best_L,
        best_H,
        best_t,
        best_mass
    ) = genetic_algorithm()

    print("Лучшая скорость V:", best_last_V)
    print("Тяга P:", best_last_P)
    print("Давление на срезе сопла:", best_p_soplar)
    print("Угол тетта (град):", best_tetta)
    print("Дальность L:", best_L)
    print("Высота H:", best_H)
    print("Время t:", best_t)
    print("Масса:", best_mass)
    print("Оптимальные значения:", best_chromosome)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
    #сохранение в бд
    save_results_to_db((best_last_V, best_last_P, best_p_soplar, best_tetta, best_L, best_H, best_t, best_mass,
        str(best_chromosome), POPULATION_SIZE, GENERATIONS, MUTATION_RATE, elapsed_time, str(PARAM_BOUNDS)))
    view_all_results()


