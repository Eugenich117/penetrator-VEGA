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
# мат модель из книжки воронцова упрощенная



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
r1 = 0.4
# d = 0.6
mass = 180
h = 125_000
mass_planet = 4.867 * 10 ** 24
Rb = 6_051_800
gravy_const = 6.67 * 10 ** (-11)
g = 8.87
dt = 0.01
tetta = -9
V = np.float64(11000)  # Используем тип данняых float64
gamma = np.float64(0)  # Используем тип данных float64
x = 0
y = 0
plotnost = []; CX = []

cToDeg = 180 / m.pi
cToRad = m.pi / 180

R = Rb + h
dV = 0
d_list = [4, 2, 1, 0.8, 0.4]

def dV_func(initial):
    S = initial['S']
    R = initial['R']
    Cxa = initial['Cxa']
    ro = initial['ro']
    V = initial['V']
    tetta = initial['tetta']
    mass = initial['mass']
    #dV = ((-1 / (2 * Px)) * Cxa * ro * V ** 2 - ((gravy_const*mass_planet)/R**2) * scipy.special.sindg(tetta)) * dt # ОСНОВНАЯ МОДЕЛЬ КОСЕНКОВОЙ
    dV = ((-mass * ((gravy_const * mass_planet) / R ** 2) * m.sin(tetta) - (0.5 * ro * V ** 2 * Cxa * S))) / mass
    return dV, 'V'

def dL_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    dL = V * Rb/R * m.cos(tetta)
    return dL, 'L'

def dtetta_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    R = initial['R']
    dtetta = ((-((gravy_const * mass_planet) / R ** 2) * m.cos(tetta)) / V + (V / R))
    #dtetta = ( ((V ** 2 - ((gravy_const*mass_planet)/R**2) * R) / (V * R)) * scipy.special.cosdg(tetta)) * dt
    return dtetta, 'tetta'

def dR_func(initial):
    V = initial['V']
    tetta = initial['tetta']
    dR = (V * m.sin(tetta))
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


#(i, napor, TETTA, X, Y, V_MOD, T, PX, nx, acceleration) # передача листов для результатов в явном виде в функцию
def compute_trajectory(i, equations, dx, pipe_conn):
    print(f"поток {i} запущен")
    t = 0
    d = d_list[i]
    S = (m.pi * d ** 2) / 4
    V, tetta, R, L = np.float64(11_000), -9 * cToRad, Rb + h, 0
    initial = {}
    initial['S'] = S
    initial['mass'] = mass

    local_TETTA = []; local_X = []; local_Y = []; local_V_MOD = []; local_T = []; local_napor = []; local_nx = []
    local_PX = []; local_acceleration = []

    while R >= Rb:
        V_sound = v_sound(R - Rb)
        ro = Get_ro(R - Rb)
        Cxa = Cx(V, V_sound)
        Px = mass / Cxa * S
        # V, tetta, R, L = runge_kutta_6(S, L, Cxa, ro, V, tetta, R, dt)
        initial.update({'tetta': tetta, 'Cxa': Cxa, 'ro': ro, 'L': L, 'V': V, 'R': R})
        values = runge_kutta_4(equations, initial, dt, dx)
        V = values[0]
        L = values[1]
        tetta = values[2]
        R = values[3]
        t += dt

        local_TETTA.append(tetta * cToDeg)
        local_X.append(L)
        local_Y.append(R - Rb)
        local_V_MOD.append(V)
        local_T.append(t)
        local_napor.append(0.5 * ro * V ** 2)
        local_nx.append((0.5 * S * Cxa * ro * V ** 2) / (mass * ((gravy_const * mass_planet) / R ** 2)))
        local_PX.append(Px)
        #print(f"Process {i} finished, data: TETTA={TETTA[i]}\n, X={X[i]}\n, Y={Y[i]}\n")  # Вывод для проверки данных

    print(f'V = {V:.3f}, tetta = {tetta * cToDeg:.3f}, L = {L:.3f}, H = {(R-Rb):.3f}, t = {t:.3f}')

    for j in range(1, len(local_V_MOD)):
        derivative_value = (local_V_MOD[j] - local_V_MOD[j - 1]) / dt
        local_acceleration.append(derivative_value)
    result = (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX, local_acceleration)
    pipe_conn.send(result)  # Передаем данные
    pipe_conn.close()  # Закрываем трубу
    #queue.put(result, block=False)



if __name__ == '__main__':

    dx = ['V', 'L', 'tetta', 'R']
    equations = [dV_func, dL_func, dtetta_func, dR_func]
    #with multiprocessing.Manager() as manager:

        # Создаем списки через менеджера
    '''manager = multiprocessing.Manager()
    acceleration = manager.list([[] for _ in range(5)])
    napor = manager.list([manager.list() for _ in range(5)])
    TETTA = manager.list([manager.list() for _ in range(5)])
    X = manager.list([manager.list() for _ in range(5)])
    Y = manager.list([manager.list() for _ in range(5)])
    T = manager.list([manager.list() for _ in range(5)])
    PX = manager.list([manager.list() for _ in range(5)])
    nx = manager.list([manager.list() for _ in range(5)])
    V_MOD = manager.list([manager.list() for _ in range(5)])'''
    '''manager = multiprocessing.Manager()
    acceleration = manager.list()
    napor = manager.list()
    TETTA = manager.list()
    X = manager.list()
    Y = manager.list()
    T = manager.list()
    PX = manager.list()
    nx = manager.list()
    V_MOD = manager.list()'''
    acceleration = ([[] for _ in range(5)])
    napor = ([[] for _ in range(5)])
    TETTA = ([[] for _ in range(5)])
    X = ([[] for _ in range(5)])
    Y = ([[] for _ in range(5)])
    T = ([[] for _ in range(5)])
    PX = ([[] for _ in range(5)])
    nx = ([[] for _ in range(5)])
    V_MOD = ([[] for _ in range(5)])

    queue = Queue(maxsize=100)
    processes = []
    parent_conns = []
    child_conns = []

    for i in range(5):
        parent_conn, child_conn = Pipe()  # Создаем пару для каждого процесса
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        p = multiprocessing.Process(target=compute_trajectory, args=(i, equations, dx, child_conn))
        p.start()
        processes.append(p)

    '''results = []
    while not queue.empty():
        results.append(queue.get())'''

    # Обработка результатов
    for i in range(5):
        result = parent_conns[i].recv()
        (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
             local_acceleration) = result
        TETTA[i] = local_TETTA
        X[i] = local_X
        Y[i] = local_Y
        V_MOD[i] = local_V_MOD
        T[i] = local_T
        napor[i] = local_napor
        nx[i] = local_nx
        PX[i] = local_PX
        acceleration[i] = local_acceleration

    for i in range(5):
        plt.plot(X[i], Y[i], label=f'Вариант {i+1}')
    plt.title('Траектории')
    plt.xlabel('Дальность, м')
    plt.ylabel('Высота, м')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        plt.plot(T[i], Y[i], label=f'Вариант {i+1}')
    plt.title('Зависимость высоты от времени')
    plt.xlabel('Время, с')
    plt.ylabel('Высота, м')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        plt.plot(T[i], V_MOD[i], label=f'Вариант {i+1}')
    plt.title('Зависимость модуля скорости от времени')
    plt.xlabel("Время, c")
    plt.ylabel('Модуль скорости, м/с')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        plt.plot(Y[i], V_MOD[i], label=f'Вариант {i+1}')
    plt.title('Зависимость модуля скорости от высоты')
    plt.xlabel("Высота, м")
    plt.ylabel('Модуль скорости, м/с')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        plt.plot(T[i], TETTA[i], label=f'Вариант {i+1}')
    plt.title('Зависимость угла входа от времени')
    plt.xlabel('Время, c')
    plt.ylabel('TETTA, град')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        plt.plot(T[i], napor[i], label=f'Вариант {i+1}')
    plt.title('Зависимость скоростного напора от времени')
    plt.xlabel('Время, с')
    plt.ylabel('Скоростной напор, Па')
    plt.legend()
    plt.grid(True)
    plt.show()

    #T.pop()# Убираем последний элемент из списка времени
    for i in range(5):
        T[i].pop()
        plt.plot(T[i], acceleration[i], label=f'Вариант {i+1}')
    plt.title('Зависимость ускорения от времени')
    plt.xlabel('Время, с')
    plt.ylabel('Ускорение м/с^2')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        nx[i].pop()
        plt.plot(T[i], nx[i], label=f'Вариант {i + 1}')
    plt.title('Зависимость перегрузки от времени')
    plt.xlabel('Время, с')
    plt.ylabel('Перегрузка, g')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(5):
        PX[i].pop()
        plt.plot(T[i], PX[i], label=f'Вариант {i + 1}')
    plt.title('Зависимость давления на мидель от времени')
    plt.xlabel('Время, с')
    plt.ylabel('Px, кг/м^2')
    plt.legend()
    plt.grid(True)
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)