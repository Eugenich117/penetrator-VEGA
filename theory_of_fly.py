import numpy as np
import time
import math as m
import matplotlib.pyplot as plt
from icecream import ic
import scipy
import bisect
import random
from scipy.integrate import quad
from findiff import FinDiff
import numpy as np
# мат модель из книжки воронцова упрощенная

r1 = 0.4
d = 1
#L = 0.53


mass = 662
h = 80_000
mass_planet = 5.9742*10**24
Rb = 6_371_100
gravy_const = 6.67*10**(-11)
g = 9.80665

S = (m.pi * d ** 2)/4
L = 0
dt = 0.01
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
    y = [9.68064e-09, 1.15819e-08, 1.39015e-08, 1.67199e-08, 2.01267e-08, 2.44041e-08,
         3.08814e-08, 3.96418e-078, 5.17245e-08, 6.87683e-08, 9.34035e-08, 1.30060e-07, 1.86514e-07,
         2.75614e-07, 3.89128e-07, 5.54951e-07, 7.99924e-07, 1.16617e-06, 1.67840e-06, 2.39477e-06, 3.41817e-06,
         4.88072e-06, 6.95782e-06, 9.69458e-06, 1.34175e-05, 1.84580e-05, 2.52385e-05, 3.43108e-05, 6.23730e-05,
         8.28284e-05, 1.09169e-04, 1.42957e-04, 1.86050e-04, 2.40709e-04, 3.09676e-04, 3.96263e-4, 5.04448e-04,
         6.39001e-04, 8.05613e-04, 1.02687e-03, 1.31669e-03, 1.71414e-03, 2.25884e-03, 2.99475e-03, 3.99566e-03,
         5.36653e-03, 7.25789e-03, 9.88736e-03, 1.35551e-02, 1.57915e-02, 1.84101e-02, 2.14783e-02, 2.50762e-02,
         2.92982e-02, 3.42565e-02, 4.00837e-02, 4.69377e-02, 5.50055e-02, 6.45096e-02, 7.57146e-02, 8.89097e-02,
         1.03995e-01, 1.21647e-01, 1.42301e-01, 1.66470e-01, 1.94755e-01, 2.27855e-01, 2.66595e-01, 3.11937e-01,
         3.64801e-01, 4.13510e-01, 4.67063e-01, 5.25786e-01, 5.90018e-01, 6.6011e-01, 7.36429e-01, 8.19347e-01,
         9.09254e-01, 1.006550, 1.111660, 1.225000]
    ro = newton_interpolation_ro(x, y, R /1000)
    return ro


def Cx(xi, V_sound):
    x = [0, 0, 0.2, 0.4, 0.6, 0.1, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    y = [0.75, 0.8, 0.9, 1.1, 1.3, 1.45, 1.51, 1.55, 1.6, 1.7, 1.8, 1.78, 1.75, 1.7, 1.65, 1.6, 1.55, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52,
         1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52, 1.52]
    return newton_interpolation(x, y, xi/V_sound)


def v_sound(R):
    x = [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86,
         84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
         31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
         2, 1, 0]
    y = [174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216,
         218, 220, 222, 224, 226, 228, 230, 232,234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260,
         262, 264, 266, 268, 270, 272,274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304,
         306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342]
    return newton_interpolation_ro(x, y, R /1000)


start_time = time.time()

x = 0
y = 0
PX = []
nx = []
ny = []
acceleration = []
napor = []
TETTA = []
X = []
Y = []
V_MOD = []
T = []
Qk = []
Tomega = []
Quantitiy_warm = []
R = Rb + h
cToDeg = 180 / m.pi
cToRad = m.pi / 180


def dV_func(initial):
    S = initial['S']
    R = initial['R']
    Cxa = initial['Cxa']
    ro = initial['ro']
    V = initial['V']
    tetta = initial['tetta']
    mass = initial['mass']
    #dV = ((-1 / (2 * Px)) * Cxa * ro * V ** 2 - ((gravy_const*mass_planet)/R**2) * scipy.special.sindg(tetta)) * dt # ОСНОВНАЯ МОДЕЛЬ КОСЕНКОВОЙ
    dV = ((-mass * (g * Rb ** 2 / R ** 2) * m.sin(tetta) - (0.5 * ro * V ** 2 * Cxa * S))) / mass
    return dV, 'V'


def dL_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    dL = V * Rb / R * m.cos(tetta)
    return dL, 'L'


def dtetta_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    R = initial['R']
    mass = initial['mass']
    dtetta = ((0.5 * ro * V ** 2 * Cya * S) / (mass *V)) - ((g * Rb ** 2 / R ** 2) / V) * m.cos(tetta) + ((V * m.cos(tetta) / R))
    #dtetta = ((-(g * Rb ** 2 / R ** 2) * m.cos(tetta)) / V + (V / R))
    #dtetta = ( ((V ** 2 - ((gravy_const*mass_planet)/R**2) * R) / (V * R)) * scipy.special.cosdg(tetta)) * dt
    return dtetta, 'tetta'


def dR_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    dR = (V * m.sin(tetta))
    return dR, 'R'


def qk_func(initial):
    ro = initial['ro']
    V = initial['V']
    dqk = ((1.318 * 10 ** 5) / m.sqrt(0.5)) * ((ro / 1.2255) ** 0.5) * ((V / 7910) ** 3.25)
    return dqk, 'qk'
    #return ((1.318 * 10 ** 5) / m.sqrt(0.5)) * m.sqrt(ro / 1.2255) * (V / 7910) ** 3.25


def quantity_func(qk):
    return qk


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
tetta = -0.034  # * (m.pi / 180)
V = 7600  # Используем тип данняых float64
qk = 0
initial = {}
initial['S'] = S
initial['mass'] = mass
dx = ['V', 'L', 'tetta', 'R', 'qk']
equations = [dV_func, dL_func, dtetta_func, dR_func, qk_func]
#tetta *= cToRad # вот это надо, если тетта в градусах
while R >= Rb:
    ro = Get_ro(R - Rb)
    V_sound = v_sound(R - Rb)
    Cxa = 0.7 #Cx(V, V_sound)
    Px = mass / Cxa * S
    xd = 0.06
    Cya = 0.025 # для баллистического сделать 0
    K = 0.15 # для баллистического сделать 0
    gamma = 0.009 # для баллистического сделать 0
    alfa = (gamma/xd) * (Cxa/(Cya + Cxa)) # для баллистического сделать 0
    Cxa = Cxa * m.cos(alfa) + Cya * m.sin(alfa)
    Cya = Cxa * m.sin(alfa) + Cya * m.cos(alfa)
    initial.update({'qk': qk, 'tetta': tetta, 'Cya': Cya, 'Cxa': Cxa, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    qk = values[4]
    t += dt

    quantity_warm, error = quad(quantity_func, 0, t)

    Quantitiy_warm.append(quantity_warm)
    Tomega.append((qk/(0.8 * 5.67 * 10**(-8)))**0.25)
    Qk.append(qk/1000)
    TETTA.append(tetta * cToDeg)
    X.append(L)
    Y.append(R-Rb)
    V_MOD.append(V)
    T.append(t)
    napor.append(0.5*ro*V**2)
    ny.append((0.5 * S * Cya * ro * V ** 2) / (mass * (g * Rb ** 2 / R ** 2)))
    nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * (g * Rb ** 2 / R ** 2)))
    PX.append(Px)
    #print(f'V = {V:.3f}, tetta = {tetta:.3f}, L = {L:.3f}, H = {(R - Rb):.3f}, t = {t}, nx ={(0.5 * S * Cxa * ro * V ** 2)/(mass*((gravy_const*mass_planet)/R**2))}')

print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')

for i in range(1, len(V_MOD)):
    derivative_value = (V_MOD[i] - V_MOD[i - 1]) / dt
    acceleration.append(derivative_value)

def beautified_plot(x, y, title, xlabel, ylabel, color='tab:red', linestyle='-', linewidth=2):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)
    plt.show()

# Построение графиков
beautified_plot(X, Y, 'Траектория', 'Дальность, м', 'Высота, м')
beautified_plot(T, Y, 'Зависимость высоты от времени', 'Время, с', 'Высота, м')
beautified_plot(T, V_MOD, 'Зависимость модуля скорости от времени', 'Время, с', 'Модуль скорости, м/с')
beautified_plot(T, TETTA, 'Зависимость угла входа от времени', 'Время, с', 'TETTA, град')
beautified_plot(T, napor, 'Зависимость скоростного напора от времени', 'Время, с', 'Скоростной напор, Па')
beautified_plot(T, nx, 'Зависимость продольной перегрузки от времени', 'Время, с', 'Перегрузка, g')
beautified_plot(T, ny, 'Зависимость поперечной перегрузки от времени', 'Время, с', 'Перегрузка, g')
beautified_plot(T, PX, 'Зависимость давления на мидель от времени', 'Время, с', 'Px, кг/м²')
beautified_plot(T, Qk, 'Зависимость плотности конвективного теплового потока от времени', 'Время, с', 'Q, кВт/м²')
beautified_plot(T, Quantitiy_warm, 'Зависимость полного количества тепла от времени', 'Время, с', 'Q, кДж/м²')
beautified_plot(T, Tomega, 'Зависимость равновесной температуры от времени', 'Время, с', 'T, K')
'''plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
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

plt.plot(T, TETTA)
plt.title('Зависимость угла входа от времени')
plt.xlabel('Время, c')
plt.ylabel('TETTA, град')
plt.grid(True)
plt.show()

plt.plot(T, napor)
plt.title('Зависимость коростного напора от времени')
plt.xlabel('Время, с')
plt.ylabel('Скоростной напор, Па')
plt.grid(True)
plt.show()

plt.plot(T, nx)
plt.title('Зависимость продольной перегрузки от времени')
plt.xlabel('Время, с')
plt.ylabel('Перегрузка, g')
plt.grid(True)
plt.show()

plt.plot(T, ny)
plt.title('Зависимость поперечной перегрузки от времени')
plt.xlabel('Время, с')
plt.ylabel('Перегрузка, g')
plt.grid(True)
plt.show()

plt.plot(T, PX)
plt.title('Зависимость давления на мидель от времени')
plt.xlabel('Время, с')
plt.ylabel('Px, кг/м^2')
plt.grid(True)
plt.show()

plt.plot(T, Qk)
plt.title('Зависимость плотности конвективного теплового потока от времени')
plt.xlabel('Время, с')
plt.ylabel('Q, КВт/м^2')
plt.grid(True)
plt.show()

plt.plot(T, Quantitiy_warm)
plt.title('Зависимость полного количества тепла к единице поверхности КЛА от времени')
plt.xlabel('Время, с')
plt.ylabel('Q, KДж/м^2')
plt.grid(True)
plt.show()

plt.plot(T, Tomega)
plt.title('Зависимость равновесной температуры поверхности КЛА от времени')
plt.xlabel('Время, с')
plt.ylabel('T, K')
plt.grid(True)
plt.show()'''
#ic(len(acceleration), len(napor), len(X), len(Y), len(T), len(Qk), len(nx), len(ny), len(V_MOD), len(Quantitiy_warm), len(Tomega))
data = {
    "acceleration": acceleration,
    "napor": napor,
    "TETTA": TETTA,
    "X": X,
    "Y": Y,
    "T": T,
    "nx": nx,
    "ny": ny,
    "V_MOD": V_MOD,
    "Qk": Qk,
    "Quantitiy_warm": Quantitiy_warm,
    "Tomega": Tomega,
}

# Перебираем каждый массив и находим min и max
for key, values in data.items():
    if isinstance(values[0], (list, np.ndarray)):  # Проверка на вложенность
        all_values = np.concatenate(values)  # Если многомерный, объединяем
    else:
        all_values = np.array(values)  # Если одномерный, преобразуем в массив

    min_val = np.min(all_values)
    max_val = np.max(all_values)
    print(f"{key}: min = {min_val}, max = {max_val}")

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)