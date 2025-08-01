import numpy as np
import time
import traceback
from scipy.special import sindg, cosdg
import math as m
import matplotlib.pyplot as plt
import matplotlib
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
from mpl_toolkits.mplot3d import Axes3D  # нужно для 3D графика

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
r1 = 0.2
r2 = 0.8
long = 1.63
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
    Px = initial['Px']
    ro = initial['ro']
    V = initial['V']
    tetta = initial['tetta']
    Cxa = initial['Cxa']
    S = initial['S']
    dV = -1 / (2 * Px) * ro * V ** 2 - (g * Rb ** 2 / R ** 2) * m.sin(tetta)
    #dV = ((-mass * (g * Rb ** 2 / R ** 2) * m.sin(tetta) - (0.5 * ro * V ** 2 * Cxa * S))) / mass
    return dV, 'V'


def dtetta_func(initial):
    Px = initial['Px']
    ro = initial['ro']
    V = initial['V']
    K = initial['K']
    gamma = initial['gamma']
    R = initial['R']
    tetta = initial['tetta']
    V_wind = initial['V_wind']
    Cya = initial['Cya']
    S = initial['S']
    #dtetta = (1 / (2 * Px)) * ro * V * K + ((V ** 2 - (g * Rb ** 2 / R ** 2) * R) / (V * R)) * m.cos(tetta)
    dtetta=((-(g*Rb**2/R**2)*m.cos(tetta))/m.sqrt(V**2+V_wind**2)+(m.sqrt(V**2+V_wind**2)/R)) # пенетратор ВА
    #dtetta = ((0.5 * ro * V ** 2 * Cya * S) / (mass * V)) - ((g * Rb ** 2 / R ** 2) / V) * m.cos(tetta) + ((V * m.cos(tetta) / R)) #  из теории полета
    return dtetta, 'tetta'


def deps_func(initial):
    Px = initial['Px']
    ro = initial['ro']
    V = initial['V']
    K = initial['K']
    tetta = initial['tetta']
    eps = initial['eps']
    phi = initial['phi']
    R = initial['R']

    deps = - V / R * m.cos(tetta) * m.tan(phi) * m.cos(eps) # первое слагаемое в системе (1 / (2 * Px)) * ro * V * (V * K * m.sin(gamma) / m.cos(tetta))
    return deps, 'eps'


def dphi_func(initial):
    V = initial['V']
    R = initial['R']
    tetta = initial['tetta']
    eps = initial['eps']

    dphi = V / R * m.cos(tetta) * m.sin(eps) + initial['omega']
    return dphi, 'phi'


def dlam_func(initial):
    """
    Изменение долготы
    initial['V'] - скорость [м/с]
    initial['theta'] - угол наклона [рад]
    initial['eps'] - угол курса [рад]
    initial['phi'] - широта [рад]
    initial['H'] - высота [м]
    """
    V = initial['V']
    tetta = initial['tetta']
    eps = initial['eps']
    phi = initial['phi']
    R = initial['R']

    dlam = V * m.cos(tetta) * m.cos(eps) / (R * m.cos(phi))
    return dlam, 'lam'


def dR_func(initial):
    V = initial['V']
    tetta = initial['tetta']

    dR = V * m.sin(tetta)
    return dR, 'R'
'''def dV_func(initial):
    """
    Изменение скорости аппарата
    initial['ro'] - плотность атмосферы [кг/м³]
    initial['V'] - скорость аппарата [м/с]
    initial['theta'] - угол наклона траектории [рад]
    initial['phi'] - широта [рад]
    initial['eps'] - угол курса [рад]
    initial['H'] - высота [м]
    initial['mass'] - масса аппарата [кг]
    initial['S'] - площадь миделя [м²]
    initial['Cx0'] - коэффициент лобового сопротивления при α=0
    initial['Cx_alpha'] - производная коэффициента сопротивления по углу атаки
    initial['alpha'] - угол атаки [рад]
    """
    # Параметры из initial
    ro = initial['ro']  # Плотность атмосферы [кг/м³]
    V = initial['V']
    tetta = initial['tetta']
    phi = initial['phi']
    eps = initial['eps']
    mass = initial['mass']
    S = initial['S']
    Cxa = initial['Cxa']
    alfa = initial['alfa']
    R = initial['R']

    g = 8.869 * Rb ** 2 / R ** 2  # Гравитация на текущей высоте

    # Аэродинамическая сила сопротивления
    Xa = 0.5 * ro * V ** 2 * S * (Cxa * alfa ** 2)  # [Н]

    # Центробежный член от вращения планеты
    centrifugal = initial['omega'] ** 2 * R * m.cos(phi) * (m.sin(tetta) * m.cos(phi) - m.cos(tetta) * m.sin(phi) * m.sin(eps))

    dV = -g * m.sin(tetta) - Xa / mass + centrifugal
    return dV, 'V'


def dtetta_func(initial):
    """
    Изменение угла наклона траектории
    initial['ro'] - плотность атмосферы [кг/м³]
    initial['V'] - скорость [м/с]
    initial['theta'] - текущий угол наклона [рад]
    initial['phi'] - широта [рад]
    initial['eps'] - угол курса [рад]
    initial['H'] - высота [м]
    initial['mass'] - масса [кг]
    initial['S'] - площадь миделя [м²]
    initial['Cy_alpha'] - производная подъемной силы по углу атаки
    initial['alpha'] - угол атаки [рад]
    """
    # Параметры из initial
    ro = initial['ro']
    V = initial['V']
    tetta = initial['tetta']
    phi = initial['phi']
    eps = initial['eps']
    mass = initial['mass']
    S = initial['S']
    Cya = initial['Cya']
    alfa = initial['alfa']
    R = initial['R']

    g = 8.869 * Rb ** 2 / R ** 2

    # Подъемная сила
    Ya = 0.5 * ro * V ** 2 * S * Cya * alfa  # [Н]

    # Центробежный член
    centrifugal = initial['omega'] ** 2 * R * m.cos(phi) * (m.cos(tetta) * m.cos(phi) + m.sin(tetta) * m.sin(phi) * m.sin(eps))
    dtetta = (-g * m.cos(tetta) / V + Ya / (mass * V) + centrifugal / V)
    return dtetta, 'tetta'


def deps_func(initial):
    """
    Изменение угла курса
    initial['ro'] - плотность атмосферы [кг/м³]
    initial['V'] - скорость [м/с]
    initial['theta'] - угол наклона [рад]
    initial['phi'] - широта [рад]
    initial['eps'] - текущий угол курса [рад]
    initial['H'] - высота [м]
    """
    V = initial['V']
    tetta = initial['tetta']
    phi = initial['phi']
    eps = initial['eps']
    R = initial['R']

    # Члены уравнения
    rotation_term = initial['omega'] ** 2 * R * m.sin(phi) * m.cos(phi) * m.cos(eps) / (V * m.cos(tetta))
    curvature_term = -V * m.cos(tetta) * m.tan(phi) * m.sin(eps) / R

    deps = rotation_term + curvature_term
    return deps, 'eps'


def dphi_func(initial):
    """
    Изменение широты
    initial['V'] - скорость [м/с]
    initial['theta'] - угол наклона [рад]
    initial['eps'] - угол курса [рад]
    initial['H'] - высота [м]
    """
    V = initial['V']
    tetta = initial['tetta']
    eps = initial['eps']
    R = initial['R']

    dphi = V * m.cos(tetta) * m.sin(eps) / R
    return dphi, 'phi'


def dlam_func(initial):
    """
    Изменение долготы
    initial['V'] - скорость [м/с]
    initial['theta'] - угол наклона [рад]
    initial['eps'] - угол курса [рад]
    initial['phi'] - широта [рад]
    initial['H'] - высота [м]
    """
    V = initial['V']
    tetta = initial['tetta']
    eps = initial['eps']
    phi = initial['phi']
    R = initial['R']

    dlam = V * m.cos(tetta) * m.cos(eps) / (R * m.cos(phi)) + initial['omega']
    return dlam, 'lam'


def dR_func(initial):
    """
    Изменение высоты
    initial['V'] - скорость [м/с]
    initial['theta'] - угол наклона [рад]
    """
    V = initial['V']
    tetta = initial['tetta']
    dR = V * m.sin(tetta)
    return dR, 'R'
    '''


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
    k1 = {key: 0 for key in dx}
    k2 = {key: 0 for key in dx}
    k3 = {key: 0 for key in dx}
    k4 = {key: 0 for key in dx}

    # k1
    for eq in equations:
        derivative, key = eq(initial)
        if key in dx:
            k1[key] = derivative

    # k2
    state_k2 = initial.copy()
    for key in dx:
        state_k2[key] += dt / 2 * k1[key]

    for eq in equations:
        derivative, key = eq(state_k2)
        if key in dx:
            k2[key] = derivative

    # k3
    state_k3 = initial.copy()
    for key in dx:
        state_k3[key] += dt / 2 * k2[key]

    for eq in equations:
        derivative, key = eq(state_k3)
        if key in dx:
            k3[key] = derivative

    # k4
    state_k4 = initial.copy()
    for key in dx:
        state_k4[key] += dt * k3[key]

    for eq in equations:
        derivative, key = eq(state_k4)
        if key in dx:
            k4[key] = derivative

    # итоговое обновление только для переменных из dx
    new_values = []
    for key in dx:
        new_val = initial[key] + (dt / 6) * (
                k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]
        )
        new_values.append(new_val)

    return new_values


#(i, napor, TETTA, X, Y, V_MOD, T, PX, nx, acceleration) # передача листов для результатов в явном виде в функцию
def compute_trajectory(i, equations, dx, pipe_conn):
    try:
        #print(f"поток {i} запущен")
        location = 'init'  # Для отслеживания этапа выполнения
        t = 0
        d = 0.8
        S = (m.pi * d ** 2) / 4
        V, tetta, R, L = random.uniform(10_900, 11_100), random.uniform(-25, -15) * cToRad, Rb + h, 0
        print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}')
        initial = {'S': S, 'mass': mass}

        local_TETTA = []; local_X = []; local_Y = []; local_V_MOD = []; local_T = []; local_napor = []; local_nx = []
        local_PX = []; local_acceleration = []; local_v_wind = []; local_wind_angle = []; local_eps = []
        local_phi = []; local_lam = []
        Qk, lam, phi, eps = m.pi / 2, 0, 0, 0
        next_update_time = -1
        V_wind = 0
        wind_angle = 0
        while R >= Rb:
            V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
            omega = 2.9926 * 10 ** -7  # Угловая скорость вращения планеты, рад/с
            location = 'v_sound'
            V_sound = v_sound(R - Rb)
            location = 'ro'
            ro = Get_ro(R - Rb)
            location = 'Cxa'
            Cxa = Cx(V, V_sound)
            location = 'Cx_wind'
            Cxa_wind = Cx_wind(V, V_sound)

            Px = mass / Cxa * S
            Cya = 0.00156 #((2 * long * r2 * (1 + r1 / r2)) / S) * np.pi * np.cos(tetta) * np.sin(tetta) * np.cos(Qk) ** 2
            xd = 0.06
            gamma = wind_angle
            alfa = (gamma / xd) * (Cxa / (Cya + Cxa))  # 0
            Cxa = Cxa * m.cos(alfa) + Cya * m.sin(alfa)
            Cya = Cxa * m.sin(alfa) + Cya * m.cos(alfa)
            K = Cya / Cxa  # 0

            initial.update({'Px': Px, 'lam': lam, 'phi': phi, 'eps': eps, 'V_wind': V_wind, 'omega': omega, 'Cya': Cya,
              'wind_angle': wind_angle, 'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V,
               'alfa': alfa, 'gamma': gamma, 'R': R, 'K': K, 'S': S})
            location = 'equations'
            values = runge_kutta_4(equations, initial, dt, dx)
            V = values[0]
            tetta = values[1]
            eps = values[2]
            phi = values[3]
            lam = values[4]
            R = values[5]
            t += dt

            local_eps.append(eps)
            local_phi.append(phi)
            local_lam.append(lam)
            local_wind_angle.append(wind_angle)
            local_v_wind.append(V_wind)
            local_TETTA.append(tetta * cToDeg)
            #local_X.append(L)
            local_Y.append(R - Rb)
            local_V_MOD.append(V)
            local_T.append(t)
            local_napor.append(0.5 * ro * V ** 2)
            local_nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
            local_PX.append(Px)

        print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')

        '''while R >= Rb:
            V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
            omega = 2.9926 * 10 ** -7  # Угловая скорость вращения планеты, рад/с
            location = 'v_sound'
            V_sound = v_sound(R - Rb)
            location = 'ro'
            ro = Get_ro(R - Rb)
            location = 'Cxa'
            Cxa = Cx(V, V_sound)
            location = 'Cx_wind'
            Cxa_wind = Cx_wind(V, V_sound)

            Px = mass / Cxa * S
            Cya = 0.00006 #((2 * long * r2 * (1 + r1 / r2)) / S) * np.pi * np.cos(tetta) * np.sin(tetta) * np.cos(Qk) ** 2
            xd = 0.06
            gamma = wind_angle
            alfa = (gamma / xd) * (Cxa / (Cya + Cxa))  # 0
            Cxa = Cxa * m.cos(alfa) + Cya * m.sin(alfa)
            Cya = Cxa * m.sin(alfa) + Cya * m.cos(alfa)
            K = Cya / Cxa  # 0

            initial.update({'Px': Px, 'lam': lam, 'phi': phi, 'eps': eps, 'V_wind': V_wind, 'omega': omega, 'Cya': Cya,
              'wind_angle': wind_angle, 'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V,
               'alfa': alfa, 'gamma': gamma, 'R': R, 'K': K, 'S': S})
            location = 'equations'
            values = runge_kutta_4(equations, initial, dt, dx)
            V = values[0]
            tetta = values[1]
            eps = values[2]
            phi = values[3]
            lam = values[4]
            R = values[5]
            t += dt

            local_eps.append(eps)
            local_phi.append(phi)
            local_lam.append(lam)
            local_wind_angle.append(wind_angle)
            local_v_wind.append(V_wind)
            local_TETTA.append(tetta * cToDeg)
            #local_X.append(L)
            local_Y.append(R - Rb)
            local_V_MOD.append(V)
            local_T.append(t)
            local_napor.append(0.5 * ro * V ** 2)
            local_nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
            local_PX.append(Px)
            #save_to_db(i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX, local_acceleration)
            #print("sucsess")


        print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')'''

        location = 'compute_acceleration'
        for j in range(1, len(local_V_MOD)):
            derivative_value = (local_V_MOD[j] - local_V_MOD[j - 1]) / dt
            local_acceleration.append(derivative_value)
        location = 'pack_result'
        result = (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
                  local_acceleration, local_v_wind, local_wind_angle, local_eps, local_phi, local_lam)
        try:
            pipe_conn.send(result)
        except (BrokenPipeError, EOFError):
            error_info = traceback.format_exc()
            if not pipe_conn.closed:
                pipe_conn.send(('error', (location, i, error_info)))
                print(f"[ОШИБКА] Процесс {i} на этапе {location}:\n{error_info}")
                print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}')


    except Exception as e:
        try:
            # Передаем не только ошибку, но и место (location)
            if not pipe_conn.closed:
                pipe_conn.send(('error', (location, i, str(e))))
                print(f"[ОШИБКА] Процесс {i} на этапе {location}: {e}")
                print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')

        except Exception as e2:
            print(f"[ОШИБКА ПРИ ОТПРАВКЕ ОШИБКИ] Процесс {i}: {e2}")
            print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')


    finally:
        pipe_conn.close()
        gc.collect()


if __name__ == '__main__':
    iter = 10 #количество итераций
    dx = ['V', 'tetta', 'eps', 'phi', 'lam', 'R']
    equations = [dV_func, dtetta_func, deps_func, dphi_func, dlam_func, dR_func]

    acceleration = ([[] for _ in range(iter)])
    napor = ([[] for _ in range(iter)])
    TETTA = ([[] for _ in range(iter)])
    X = ([[] for _ in range(iter)])
    Y = ([[] for _ in range(iter)])
    T = ([[] for _ in range(iter)])
    PX = ([[] for _ in range(iter)])
    nx = ([[] for _ in range(iter)])
    V_MOD = ([[] for _ in range(iter)])
    V_WIND = ([[] for _ in range(iter)])
    WIND_ANGLE = ([[] for _ in range(iter)])
    EPS = ([[] for _ in range(iter)])
    PHI = ([[] for _ in range(iter)])
    LAM = ([[] for _ in range(iter)])
    processes = []
    parent_conns = []
    child_conns = []

    tasks = []

    for i in range(iter):
        parent_conn, child_conn = multiprocessing.Pipe()  # Создаем пару для каждого процесса
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        tasks.append((i, equations, dx, child_conn))  # Формируем задачи для передачи в пул

    # Функция обратного вызова, которая будет вызываться по завершении каждой задачи
    # Создаем пул процессов
    pool = multiprocessing.Pool(processes=30)

    # Запускаем задачи асинхронно с отслеживанием завершения
    try:
        # Запуск задач
        for task in tasks:
            pool.apply_async(compute_trajectory, task)

        pool.close()  # Больше задач не будет

        # Теперь безопасно читаем результаты
        for i in range(iter):
            try:
                if parent_conns[i].poll(200):  # Ждем максимум 60 секунд
                    result = parent_conns[i].recv()
                    (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
                     local_acceleration, local_v_wind, local_wind_angle, local_eps, local_phi, local_lam) = result

                    TETTA[i] = local_TETTA
                    X[i] = local_X
                    Y[i] = local_Y
                    V_MOD[i] = local_V_MOD
                    T[i] = local_T
                    napor[i] = local_napor
                    nx[i] = local_nx
                    PX[i] = local_PX
                    acceleration[i] = local_acceleration
                    WIND_ANGLE[i] = local_wind_angle
                    V_WIND[i] = local_v_wind
                    EPS[i] = local_eps
                    PHI[i] = local_phi
                    LAM[i] = local_lam
                else:
                    raise TimeoutError(f'Процесс {i} не прислал данные за отведённое время.')

            except Exception as e:
                print(f"[ОШИБКА] При получении данных от процесса {i}: {e}")

    except Exception as e:
        print(f"[КРИТИЧЕСКАЯ ОШИБКА] {e}")

    finally:
        pool.terminate()  # На всякий случай убиваем все процессы
        pool.join()  # Ждем завершения

    matplotlib.use('Qt5Agg')  # или 'TkAgg' если с Qt проблемы
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    for i in range(iter):
        ax.plot( PHI[i], LAM[i], Y[i], label=f'Вариант {i+1}')

    ax.set_title('Траектории спуска зонда-пенетратора (3D)', fontsize=14, fontname='Times New Roman')
    ax.set_xlabel('PHI (широта/долгота), рад', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('LAM (курс), рад', fontsize=12, fontname='Times New Roman')
    ax.set_zlabel('Высота, м', fontsize=12, fontname='Times New Roman')
    ax.grid(True)
    # ax.legend()  # при необходимости

    plt.tight_layout()
    plt.show()

    for i in range(iter):
        plt.plot(PHI[i], Y[i], label=f'Вариант {i+1}')
    plt.title('Траектории спуска зонда-пенетратора', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Дальность, м', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Высота, м', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        plt.plot(T[i], Y[i], label=f'Вариант {i+1}')
    plt.title('Зависимость высоты от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Высота, м', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        plt.plot(T[i], V_MOD[i], label=f'Вариант {i+1}')
    plt.title('Зависимость модуля скорости от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel("Время, c", fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'Скорость, $\frac{м}{с}$', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        plt.plot(Y[i], V_MOD[i], label=f'Вариант {i+1}')
    plt.title('Зависимость модуля скорости от высоты', fontsize=16, fontname='Times New Roman')
    plt.xlabel("Высота, м", fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'Скорость, $\frac{м}{с}$', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        plt.plot(T[i], TETTA[i], label=f'Вариант {i+1}')
    plt.title('Зависимость траекторного угла от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, c', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Траекторный угол, град', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        plt.plot(T[i], napor[i], label=f'Вариант {i+1}')
    plt.title('Зависимость скоростного напора от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'Скоростной напор, $\frac{\mathrm{кг}}{\mathrm{м} \cdot \mathrm{с}^{2}}$', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    #T.pop()# Убираем последний элемент из списка времени
    for i in range(iter):
        T[i].pop()
        plt.plot(T[i], acceleration[i], label=f'Вариант {i+1}')
    plt.title('Зависимость ускорения от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Ускорение м$^2$', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        nx[i].pop()
        plt.plot(T[i], nx[i], label=f'Вариант {i + 1}')
    plt.title('Зависимость перегрузки от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Перегрузка, g', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        PX[i].pop()
        plt.plot(T[i], PX[i], label=f'Вариант {i + 1}')
    plt.title('Зависимость давления на мидель от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'Px, $\frac{кг}{м^2}$', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        V_WIND[i].pop()
        plt.plot(T[i], V_WIND[i], label=f'Вариант {i + 1}')
    plt.title('Скорость ветра от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'м/с', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        WIND_ANGLE[i].pop()
        plt.plot(T[i], WIND_ANGLE[i], label=f'Вариант {i + 1}')
    plt.title('Угол действия ветра от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel(r'Рад/с', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(iter):
        T[i] = T[i][:3000]
        nx[i] = nx[i][:3000]
        plt.plot(T[i], nx[i], label=f'Вариант {i + 1}')
    plt.title('Зависимость перегрузки от времени', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Время, с', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Перегрузка, g', fontsize=16, fontname='Times New Roman')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    #plt.legend()
    plt.grid(True)
    plt.show()

    data = {
        "acceleration": acceleration,
        "napor": napor,
        "TETTA": TETTA,
        "X": X,
        "Y": Y,
        "T": T,
        "PX": PX,
        "nx": nx,
        "V_MOD": V_MOD,
    }

    # Перебираем каждый массив и находим min и max
    for key, values in data.items():
        all_values = np.concatenate(values)  # Объединяем все списки в один массив
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        print(f"{key}: min = {min_val}, max = {max_val}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)