import numpy as np
import time
from scipy.special import sindg, cosdg
import math as m
import matplotlib.pyplot as plt
from icecream import ic
import scipy
import random
import bisect

# мат модель из книжки воронцова упрощенная

r1 = 0.4
d = 0.92
#L = 0.53
mass = 600
h = 450_000
mass_planet = 1.8996*10**27
Rb = 69_911_800
gravy_const = 6.67*10**(-11)
g = 24.79

S = (m.pi * d ** 2)/4
L = 0
tetta = -8.3  # * (m.pi / 180)
V = np.float64(47_417)  # Используем тип данняых float64
dR = 0
dt = 0.01
dtetta = 0
t = 0.0


# Cxa = 1.3#((2*L*r2*(1+r1/r2)/S))*(m.tan(Qk)/2)*(2*m.cos(0)**2*m.sin(Qk)**2+m.sin(0))
# Cya = 0#((2*L*r2*(1+r1/r2))/S)*m.pi*m.cos(0)*m.sin(0)*m.cos(Qk)*m.cos(Qk)
# Px = mass / Cxa * S
# K = Cya / Cxa
# ic(Cxa, Cya)



def find_closest_points_ro(x, xi):
    """
    Находит шесть ближайших точек к xi в списке x.
    """
    closest_points = []
    for i in range(len(x)):
        if xi - 3 <= x[i] <= xi + 3:
            if i <= 3:
                closest_indices = list(range(6))
            elif i >= len(x) - 3:
                closest_indices = list(range(len(x) - 6, len(x)))
            else:
                closest_indices = list(range(i - 3, i + 3))
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

    x = [450, 445, 440, 435, 430, 425, 420, 415, 410, 405, 400, 395, 390, 385, 380, 375, 370,
    365, 360, 355, 350, 345, 340, 335, 330, 325, 320, 315, 310, 305, 300, 295, 290, 285, 280, 275, 270, 265, 260, 255,
    250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140,
    135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5,
    0, -5, -10, -15, -20, -20, -20, -20]

    y = [2.90291e-09, 3.21859e-09, 3.57229e-09,
    3.96906e-09, 4.41466e-09, 4.91572e-09, 5.47986e-09, 6.11584e-09, 6.83374e-09, 7.64520e-09, 8.56371e-09, 9.60484e-09,
    1.07867e-08, 1.21303e-08, 1.36601e-08, 1.54046e-08, 1.73972e-08, 1.96770e-08, 2.22898e-08, 2.52894e-08, 2.87393e-08,
    3.27144e-08, 3.73036e-08, 4.26118e-08, 4.87643e-08, 5.59101e-08, 6.42275e-08, 7.39298e-08, 8.52736e-08, 9.85678e-08,
    1.14186e-07, 1.32579e-07, 1.54299e-07, 1.80014e-07, 2.10545e-07, 2.46898e-07, 2.90313e-07, 3.42323e-07, 4.04830e-07,
    4.80202e-07, 5.71405e-07, 6.82161e-07, 8.17169e-07, 9.82383e-07, 1.18539e-06, 1.43590e-06, 1.74639e-06, 2.13304e-06,
    2.61686e-06, 3.22538e-06, 3.99482e-06, 4.97322e-06, 6.22470e-06, 7.83545e-06, 9.92223e-06, 1.26445e-05, 1.62218e-05,
    2.09593e-05, 2.72849e-05, 3.58051e-05, 4.73884e-05, 6.32929e-05, 8.53633e-05, 0.00011634, 0.000160351, 0.000223708,
    0.000316221, 0.0004534, 0.000660247, 0.000977892, 0.001154504, 0.001569439, 0.002126011, 0.002867584, 0.003848036,
    0.005133048, 0.00680114, 0.008944385, 0.011668899, 0.015095332, 0.019359783, 0.024615631, 0.03103683, 0.038823057,
    0.048206999, 0.05946387, 0.072923233, 0.088983204, 0.108127299, 0.130944361,  0.171,  0.2146, 0.24165,
    0.2687, 0.2687, 0.2687, 0.2687]

    ro = newton_interpolation_ro(x, y, R / 1000)
    return ro


def Cx(M):
    x = [0, 0, 0.2, 0.4, 0.6, 0.1, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
         67]
    y = [0.75, 0.8, 0.9, 1.1, 1.3, 1.45, 1.52, 1.55, 1.6, 1.7, 1.8, 1.78, 1.75, 1.7, 1.65, 1.6, 1.55, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52]
    return newton_interpolation(x, y, M)


def Cx_wind(M):
    x = [0, 0, 0.2, 0.4, 0.6, 0.1, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
         67]
    y = [0.15, 0.15, 0.18, 0.3, 0.38, 0.81, 0.92, 0.97, 0.995, 0.991, 0.985, 0.98, 0.975, 0.97, 0.955, 0.935, 0.925,
         0.91, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
         0.9, 0.9, 0.9, 0.9]
    return newton_interpolation(x, y, M)


def M(R):
    x = [450, 445, 440, 435, 430, 425, 420, 415, 410, 405, 400, 395, 390, 385, 380, 375, 370,
    365, 360, 355, 350, 345, 340, 335, 330, 325, 320, 315, 310, 305, 300, 295, 290, 285, 280, 275, 270, 265, 260, 255,
    250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145, 140,
    135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5,
    0, 0, 0, 0]

    y = [0.0021298, 0.0021323, 0.0021348, 0.0021373, 0.0021398, 0.0021423, 0.0021448, 0.0021473, 0.0021498, 0.0021523,
         0.0021548, 0.0021573, 0.0021598, 0.0021623, 0.0021648, 0.0021673, 0.0021698, 0.0021723, 0.0021748, 0.0021773,
         0.0021798, 0.0021823, 0.0021848, 0.0021873, 0.0021898, 0.0021923, 0.0021948, 0.0021973, 0.0021998, 0.0022023,
         0.0022048, 0.0022073, 0.0022098, 0.0022123, 0.0022148, 0.0022173, 0.0022198, 0.0022223, 0.0022248, 0.0022273,
         0.0022298, 0.0022323, 0.0022348, 0.0022373, 0.0022398, 0.0022423, 0.0022448, 0.0022473, 0.0022498, 0.0022523,
         0.0022548, 0.0022573, 0.0022598, 0.0022623, 0.0022648, 0.0022673, 0.0022698, 0.0022723, 0.0022748, 0.0022773,
         0.0022798, 0.0022823, 0.0022848, 0.0022873, 0.0022898, 0.0022923, 0.0022948, 0.0022973, 0.0022998, 0.0023023,
         0.0023048, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065,
         0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065, 0.0023065,
         0.0023065]
    return newton_interpolation(x, y, R / 1000)


start_time = time.time()

x = 0
y = 0
PX = []
nx = []
plotnost = []
acceleration = []
napor = []
TETTA = []
CX = []
X = []
Y = []
V_MOD = []
T = []
R = Rb + h
dV = 0

cToDeg = 180 / m.pi
cToRad = m.pi / 180

def wind(h, t, next_update_time, V_wind, wind_angle):
    bounds = [0, 10_000, 30_000, 50_000, 70_000, 100_000, 125_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000,
              450_000, float('inf')]
    # Функциидлявычисленияv_windвзависимостиотдиапазона
    actions = [
        lambda: random.uniform(10, 20),  # 0<h<10
        lambda: random.uniform(20, 35),  # 10<h<30
        lambda: random.uniform(35, 45),  # 30<h<50
        lambda: random.uniform(45, 60),  # 50<h<70
        lambda: random.uniform(60, 75),  # 70<h<100
        lambda: random.uniform(75, 100),  # 100<h<125
        lambda: random.uniform(100, 110),  # 125<h<150
        lambda: random.uniform(110, 125),  # 150<h<200
        lambda: random.uniform(125, 130),  # 200<h<250
        lambda: random.uniform(130, 140),  # 250<h<300
        lambda: random.uniform(140, 145),  # 300<h<350
        lambda: random.uniform(145, 147),  # 350<h<400
        lambda: random.uniform(147, 150),  # 400<h<450

        lambda: 150  #h>200
        ]
    #Находим индекс диапазона с помощью bisect
    index = bisect.bisect_right(bounds, h) - 1

    if t >= next_update_time:
        V_wind = actions[index]()  # Генерируем новую скорость ветра
        wind_angle = random.uniform(0, m.pi)  # Генерируем новый угол ветра
        wind_timer = random.uniform(0.2, 10)  # Случайный таймер
        next_update_time = t + wind_timer  # Устанавливаем время следующего обновления
    return V_wind, wind_angle, next_update_time


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
    V_wind_x = V_wind*m.sin(wind_angle)#Вдольтраектории
    U = initial['U']
    Cn = initial['Cn']
    Fn = initial['Fn']
    F = ro * (1 - (430/176)) * U * (g * Rb ** 2 / R ** 2)
    #dV = (-mass * (g * Rb**2 / R**2) * m.sin(tetta) - (0.5 * ro * V**2 * Cxa * S) + sign(V_wind_x) * (0.5 * ro * V_wind_x**2 * Cxa_wind * S)) / mass #без аэростата
    '''dV = (((ro * g * U * m.sin(tetta)) - mass * (g * Rb ** 2 / R ** 2) * m.sin(tetta) -(0.5 * ro * V ** 2 *
        (Cxa * S + Cn * Fn)) + sign(V_wind_x) * (0.5 * ro * V_wind_x**2 * (Cxa_wind * S + Cn * Fn) * S))) / mass # нормальная модель с аэростатом'''
    dV = (((ro * (1 - (420/176)) * U * (g * Rb ** 2 / R ** 2) * m.sin(tetta)) / mass - mass * (g * Rb ** 2 / R ** 2) * m.sin(tetta) -(0.5 * ro * V ** 2 *
        (Cxa * S + Cn * Fn)) + sign(V_wind_x) * (0.5 * ro * V_wind_x**2 * (Cxa_wind * S + Cn * Fn) * S))) / mass # модель для аэростата монгольфьера'''
    return dV, 'V'

def dL_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    V_wind = initial['V_wind']
    wind_angle = initial['wind_angle']

    V_wind_z = V_wind * m.cos(wind_angle)#Перпендикулярнотраектории
    dL = m.sqrt(V**2 + V_wind_z**2) * Rb / R * m.cos(tetta)
    return dL, 'L'

def dtetta_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    R = initial['R']
    V_wind = initial['V_wind']
    U = initial['U']
    ro = initial['ro']

    '''dtetta = (((ro * (1 - (330 / 176)) * U * (g * Rb ** 2 / R ** 2) * m.cos(tetta)) - (g * Rb ** 2 / R ** 2) * m.cos(tetta)) / m.sqrt(V ** 2 + V_wind ** 2) + (
                m.sqrt(V ** 2 + V_wind ** 2) / R)) # модель для аэростата монгольфьера'''
    dtetta = (((ro * (1 - (420/176)) * U * (g * Rb ** 2 / R ** 2) * m.cos(tetta)) / mass - (g * Rb**2 / R**2) * m.cos(tetta)) / m.sqrt(V**2 + V_wind**2) + (m.sqrt(V**2 + V_wind**2) / R))# нормальная модель с аэростатом
    #dtetta=(((V**2-((gravy_const*mass_planet)/R**2)*R)/(V*R))*scipy.special.cosdg(tetta))*dt
    return dtetta, 'tetta'

def dR_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    V_wind = initial['V_wind']
    wind_angle = initial['wind_angle']

    V_wind = V_wind * m.cos(wind_angle)#Вдольтраектории
    dR = (m.sqrt(V**2 + V_wind**2) * m.sin(tetta))
    return dR, 'R'



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

initial = {}
initial['S'] = S
initial['mass'] = mass
dx = ['V', 'L', 'tetta', 'R']
equations = [dV_func, dL_func, dtetta_func, dR_func]
tetta *= cToRad
mass = 600
t = 0
d = 0.92
S = (m.pi * d ** 2) / 4
V, tetta, R, L = 47_000,  -9.0 * cToRad, Rb + h, 0
print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}')
initial = {}
Cn, Fn, U = 0, 0, 0
initial['S'] = S
initial['mass'] = mass
V_wind = 0
wind_angle = 0
next_update_time = -1
while R >= Rb + 15_000:
    mah = M(R - Rb)
    ro = Get_ro(R - Rb)
    Cxa = Cx(mah)
    Px = mass / Cxa * S
    V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
    Cxa_wind = Cx_wind(mah)
    initial.update({'S': S, 'U': U, 'Px': Px, 'Cn': Cn, 'Fn': Fn, 'V_wind': V_wind, 'wind_angle': wind_angle,
        'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt

    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R - Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5 * ro * V ** 2)
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
    PX.append(Px)
    #print(f"Process {i} finished, data: TETTA={TETTA[i]}\n, X={X[i]}\n, Y={Y[i]}\n")  # Вывод для проверки данных
print(f'1) V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')

mass, Cn, Fn, U = 480, 0.35, 50, 0
while R >= Rb + 14_300:
    mah = M(R - Rb)
    ro = Get_ro(R - Rb)
    Cxa = Cx(mah)
    Px = mass / Cxa * S
    V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
    Cxa_wind = Cx_wind(mah)
    initial.update({'S': S, 'U': U, 'Px': Px, 'Cn': Cn, 'Fn': Fn, 'V_wind': V_wind, 'wind_angle': wind_angle,
        'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt

    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R - Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5 * ro * V ** 2)
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
    PX.append(Px)
print(f'2) V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}')

Cn, Fn, U = 0.65, 150, 0
mass = 520
while R >= Rb + 13_000:
    mah = M(R - Rb)
    ro = Get_ro(R - Rb)
    Cxa = Cx(mah)
    Px = mass / Cxa * S
    V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
    Cxa_wind = Cx_wind(mah)
    initial.update({'S': S, 'U': U, 'Px': Px, 'Cn': Cn, 'Fn': Fn, 'V_wind': V_wind, 'wind_angle': wind_angle,
        'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt

    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R - Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5 * ro * V ** 2)
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
    PX.append(Px)
print(f'3) V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}')

Cn, Fn, U = 0.85, 300, 0
while R >= Rb + 12_000:
    mah = M(R - Rb)
    ro = Get_ro(R - Rb)
    Cxa = Cx(mah)
    Px = mass / Cxa * S
    V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
    Cxa_wind = Cx_wind(mah)
    initial.update({'S': S, 'U': U, 'Px': Px, 'Cn': Cn, 'Fn': Fn, 'V_wind': V_wind, 'wind_angle': wind_angle,
        'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt

    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R - Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5 * ro * V ** 2)
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
    PX.append(Px)
print(f'4) V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}')

Cn, Fn, U = 0.9, 450, 0
mass = 450
while R >= Rb:
    mah = M(R - Rb)
    ro = Get_ro(R - Rb)
    Cxa = Cx(mah)
    Px = mass / Cxa * S
    V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
    Cxa_wind = Cx_wind(mah)
    initial.update({'S': S, 'U': U, 'Px': Px, 'Cn': Cn, 'Fn': Fn, 'V_wind': V_wind, 'wind_angle': wind_angle,
        'tetta': tetta, 'Cxa': Cxa, 'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt

    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R - Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5 * ro * V ** 2)
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
    PX.append(Px)
print(f'5) V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}')

'''Cn, Fn, U = 1.1, 450, 40**3 * 0.55
while t <= 1550:
    mah = M(R - Rb)
    ro = Get_ro(R - Rb)
    Cxa = 1.28 # Cx(mah)
    Px = mass / Cxa * S
    V_wind, wind_angle, next_update_time = wind(R - Rb, t, next_update_time, V_wind, wind_angle)
    Cxa_wind = Cx_wind(mah)
    initial.update({'S': S, 'U': U, 'Px': Px, 'Cn': Cn, 'Fn': Fn, 'V_wind': V_wind, 'wind_angle': wind_angle, 'tetta': tetta, 'Cxa': Cxa,
        'Cxa_wind': Cxa_wind, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt

    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R - Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5 * ro * V ** 2)
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
    PX.append(Px)
print(f'6) V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t:.3f}, U = {U:.3f}')'''


for i in range(1, len(V_MOD)):
    derivative_value = (V_MOD[i] - V_MOD[i - 1]) / dt
    acceleration.append(derivative_value)

plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.plot(X, Y)
plt.title('Траектория')
plt.xlabel('Дальность, м')
plt.ylabel('Высота, м')
plt.grid(True)
plt.show()

plt.plot(T, Y)
plt.title('Зависимость высоты от времени')
plt.xlabel('Время, с')
plt.ylabel('Высота, м')
plt.grid(True)
plt.show()

plt.plot(T, V_MOD)
plt.title('Зависимость модуля скорости от времени')
plt.xlabel("Время, c")
plt.ylabel('Модуль скорости, м/с')
plt.grid(True)
plt.show()

plt.plot(Y, V_MOD)
plt.title('Зависимость модуля скорости от высоты')
plt.xlabel("Модуль скорости, м/с")
plt.ylabel('Высота, м')
plt.grid(True)
plt.show()

'''plt.plot( plotnost, V_MOD)
plt.title('Зависимость модуля скорости от плотности')
plt.xlabel('Плотность кг/м^3')
plt.ylabel('Модуль скорости, м/с')
plt.grid(True)
plt.show()'''

plt.plot(T, TETTA)
plt.title('Зависимость угла входа от времени')
plt.xlabel('Время, c')
plt.ylabel('TETTA, град')
plt.grid(True)
plt.show()

'''plt.plot(LAM, PHI)
plt.title('Траектория')
plt.xlabel('Широта, град')
plt.ylabel('Долгота, град')
plt.grid(True)
plt.show()'''

'''plt.plot(T, CX)
plt.title('Аэродинамические коэффициенты')
plt.xlabel('Время, с')
plt.ylabel('Сxa')
plt.grid(True)
plt.show()'''

plt.plot(T, napor)
plt.title('Зависимость коростного напора от времени')
plt.xlabel('Время, с')
plt.ylabel('Скоростной напор, Па')
plt.grid(True)
plt.show()

T.pop()
plt.plot(T, acceleration)
plt.title('Зависимость ускорения от времени')
plt.xlabel('Время, с')
plt.ylabel('Ускорение м/с^2')
plt.grid(True)
plt.show()

nx.pop()
plt.plot(T, nx)
plt.title('Зависимость перегрузки от времени')
plt.xlabel('Время, с')
plt.ylabel('Перегрузка, g')
plt.grid(True)
plt.show()

PX.pop()
plt.plot(T, PX)
plt.title('Зависимость давления на мидель от времени')
plt.xlabel('Время, с')
plt.ylabel('Px, кг/м^2')
plt.grid(True)
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)