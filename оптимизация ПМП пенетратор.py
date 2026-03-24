#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизация начальных условий входа зонда-пенетратора (Венера).
Управление ОТСУТСТВУЕТ — чисто баллистическое неуправляемое движение.
Оптимизируемые параметры: V0 (скорость входа), theta0 (угол входа).
Критерий: J = lambda_Q * Q(T) - lambda_V * V(T) → min
  (минимум суммарного теплового нагрева и максимум скорости при контакте)
"""

import math
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import warnings
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')


def _select_font():
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ('Times New Roman', 'DejaVu Serif', 'Georgia', 'serif'):
        if name in available:
            return name
    return 'serif'


matplotlib.rcParams.update({'font.family': _select_font(), 'font.size': 12})


def _mp_ctx():
    if sys.platform in ('win32', 'darwin'):
        return _mp.get_context('spawn')
    return _mp.get_context('fork')


# ─── Параметры Венеры и аппарата ──────────────────────────────────────────────
Rb = 6_051_800.0          # радиус Венеры, м
G0 = 8.87                 # ускорение свободного падения у поверхности, м/с²
mass = 120.0              # масса аппарата, кг

d_body = 0.27             # диаметр корпуса, м
Qk = 15.0 * math.pi / 180.0          # полуугол конуса носовой части, рад
r1 = 0.01                 # радиус кончика носа, м
r2 = d_body / 2.0         # радиус корпуса, м
Long_penetrator = (r2 - r1) / math.tan(Qk)   # длина конической части, м
S = math.pi * r2 ** 2     # характерная площадь, м²
r_nose = r1               # радиус кривизны носа для теплового расчёта, м

# Параметры стенки (для конвективного нагрева от атмосферы)
h_conv = 80.0             # коэффициент теплообмена, Вт/(м²·К)
T_wall = 600.0            # температура стенки, К
R_CO2 = 188.9             # газовая постоянная CO2, Дж/(кг·К)

# Весовые коэффициенты критерия оптимальности
lambda_V = 1.0            # вес скорости приземления (максимизировать)
lambda_Q  = 5.0e-6        # вес суммарного нагрева (минимизировать)

# Параметры интегрирования
DT = 0.2                  # базовый шаг, с
DT_FINE = 0.02            # уточнённый шаг на малых высотах (<3 км), с
T_MAX = 3500.0            # максимальное время полёта, с
DEG = math.pi / 180.0

# ─── Фиксированная высота входа и базовая точка ───────────────────────────────
# ВНИМАНИЕ: h0 НЕ является оптимизируемым параметром.
# Оптимизируются только V0 и theta0.
h0_ref     = 125_000.0   # высота входа в атмосферу (фиксирована), м
V0_ref     = 11_000.0    # базовая скорость входа, м/с
theta0_ref = -19.0 * DEG  # базовый угол входа, рад

# ─── Диапазоны ТОЛЬКО для V0 и theta0 ────────────────────────────────────────
BOUNDS = [
    (9_000.0,  11_500.0),          # V0, м/с
    (-30.0 * DEG, -5.0 * DEG),     # theta0, рад
]

# ─── Таблицы атмосферы Венеры ─────────────────────────────────────────────────
_H = [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102,
      100,  98,  96,  94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74,  72,
       70,  68,  66,  64,  62,  60,  58,  56,  54,  52,  50,  48,  46,  44,  42,
       40,  38,  36,  34,  32,  31,  30,  29,  28,  27,  26,  25,  24,  23,  22,
       21,  20,  19,  18,  17,  16,  15,  14,  13,  12,  11,  10,   9,   8,   7,
        6,   5,   4,   3,   2,   1,   0]

_RHO = [3.97200000e-08, 7.890e-7, 1.35931821e-06, 1.77164551e-06, 2.30904567e-06,
        3.00945751e-06, 3.92232801e-05, 5.11210308e-05, 6.66277727e-05, 8.68382352e-05,
        1.13179216e-04, 1.47510310e-04, 1.92255188e-04, 2.50572705e-04, 3.26579903e-04,
        1.347e-4, 0.0002, 0.0004, 0.0007, 0.0012, 0.0019, 0.0031, 0.0049, 0.0077,
        0.0119, 0.0178, 0.0266, 0.0393, 0.0578, 0.0839,
        0.1210, 0.1729, 0.2443, 0.3411, 0.4694, 0.6289,
        0.8183, 1.0320, 1.2840, 1.5940, 1.9670, 2.4260, 2.9850, 3.6460, 4.4040,
        5.2760, 6.2740, 7.4200, 8.7040, 9.4060, 10.1500, 10.9300, 11.7700, 12.6500,
        13.5900, 14.5700, 15.6200, 16.7100, 17.8800, 19.1100, 20.3900, 21.7400,
        23.1800, 24.6800, 26.2700, 27.9500, 29.7400, 31.6000, 33.5400, 35.5800,
        37.7200, 39.9500, 42.2600, 44.7100, 47.2400, 49.8700, 52.6200, 55.4700,
        58.4500, 61.5600, 64.7900, 66.15]

_VS = [174, 176, 178, 180, 182, 185, 186, 187, 190, 193, 195, 196, 198, 199, 201,
       203, 205, 206, 208.0, 208.0, 209.0, 212.2, 215.4, 218.6, 221.8, 225.0,
       228.2, 231.4, 234.6, 237.8, 241.0, 244.0, 247.0, 250.0, 253.0, 256.0,
       263.2, 270.4, 277.6, 284.8, 292.0, 296.8, 301.6, 306.4, 311.2, 316.0,
       321.2, 326.4, 331.6, 336.8, 339.4, 342.0, 344.6, 347.2, 349.8, 352.4,
       355.0, 357.4, 359.8, 362.2, 364.6, 367.0, 369.4, 371.8, 374.2, 376.6,
       379.0, 381.0, 383.0, 385.0, 387.0, 389.0, 391.2, 393.4, 395.6, 397.8,
       400.0, 402.0, 404.0, 406.0, 408.0, 410.0]

_PRES = [0.0019907, 0.005, 0.10, 0.30, 0.50, 0.70, 0.90, 1.10, 1.30, 1.50, 1.70,
         1.90, 2.10, 2.30, 2.50, 2.66, 4.45, 7.519, 12.81, 20.0, 40.0, 60.0,
         110.0, 170.0, 280.0, 450.0, 700.0, 1080.0, 1650.0, 2480.0, 3690.0, 5450.0,
         7970.0, 11560.0, 16590.0, 23570.0, 33060.0, 45590.0, 61600.0, 81670.0,
         106600.0, 137500.0, 175600.0, 222600.0, 280200.0, 350100.0, 434200.0,
         534600.0, 653700.0, 794000.0, 872900.0, 958100.0, 1050000.0, 1149000.0,
         1256000.0, 1370000.0, 1493000.0, 1625000.0, 1766000.0, 1917000.0,
         2079000.0, 2252000.0, 2436000.0, 2633000.0, 2843000.0, 3066000.0,
         3304000.0, 3557000.0, 3826000.0, 4112000.0, 4416000.0, 4739000.0,
         5081000.0, 5444000.0, 5828000.0, 6235000.0, 6665000.0, 7120000.0,
         7601000.0, 8109000.0, 8645000.0, 9210000.0]


# ─── Интерполяция (4-точечный полином Ньютона) ────────────────────────────────
def _n4(x_tbl, y_tbl, xi):
    """Интерполяция по 4 ближайшим точкам методом Ньютона."""
    n = len(x_tbl)
    idx = None
    for i in range(n):
        if xi - 2 <= x_tbl[i] <= xi + 2:
            if i <= 2:
                idx = list(range(4))
            elif i >= n - 2:
                idx = list(range(n - 4, n))
            else:
                idx = list(range(i - 2, i + 2))
            break
    if idx is None:
        idx = list(range(4)) if xi > x_tbl[0] else list(range(n - 4, n))
    xp = [x_tbl[k] for k in idx]
    yp = [y_tbl[k] for k in idx]
    for j in range(1, 4):
        for i in range(3, j - 1, -1):
            if xp[i] != xp[i - j]:
                yp[i] = (yp[i] - yp[i - 1]) / (xp[i] - xp[i - j])
    res = yp[3]
    for i in range(2, -1, -1):
        res = res * (xi - xp[i]) + yp[i]
    return res


def Get_ro(h_m):
    """Плотность атмосферы Венеры, кг/м³."""
    h_km = max(0.0, min(130.0, h_m / 1000.0))
    return max(_n4(_H, _RHO[:], h_km), 1e-14)


def v_sound(h_m):
    """Скорость звука, м/с."""
    h_km = max(0.0, min(130.0, h_m / 1000.0))
    return max(_n4(_H, _VS[:], h_km), 1.0)


def pressure_atm(h_m):
    """Атмосферное давление, Па."""
    h_km = max(0.0, min(130.0, h_m / 1000.0))
    return max(_n4(_H, _PRES[:], h_km), 0.0)


def T_atm(h_m):
    """Температура атмосферы через уравнение состояния идеального газа, К."""
    ro = Get_ro(h_m)
    return pressure_atm(h_m) / (ro * R_CO2)


# ─── Аэродинамика: коэффициент лобового сопротивления ────────────────────────
def Cx_func(V, h_m):
    """
    Cx в зависимости от числа Маха.
    M < 0.8 — дозвук (ньютоновский + трение)
    0.8–1.2 — трансзвук (плавный переход)
    1.2–5   — сверхзвук (волновое + трение)
    M > 5   — гиперзвук (ньютоновский с поправкой на сжатие)
    """
    vs = v_sound(h_m)
    M = V / vs
    if M < 0.8:
        Cx_form = 0.8 * math.sin(Qk) ** 2
        Cx_friction = 0.074 / (1e6 ** 0.2) * (Long_penetrator / (2.0 * r2))
        return min((Cx_form + Cx_friction) * (0.7 + 0.3 * (M / 0.8)), 1.0)
    elif M < 1.2:
        Cx_sub = 0.8 * math.sin(Qk) ** 2
        Cx_sup = 2.0 * math.sin(Qk) ** 2 / math.sqrt(M ** 2 - 0.5)
        t = (M - 0.8) / 0.4
        peak = 1.0 + 2.0 * math.sin(math.pi * (M - 0.9) / 0.2) ** 2
        return min(Cx_sub + (Cx_sup - Cx_sub) * t * peak, 2.5)
    elif M < 5.0:
        beta = math.sqrt(M ** 2 - 1.0)
        Cp_cone = 2.0 * math.sin(Qk) ** 2 / beta
        Cx_friction = 0.03 / (M * 1e6) ** 0.2 * (Long_penetrator / (2.0 * r2))
        return min(Cp_cone * math.sin(Qk) + Cx_friction, 1.5)
    else:
        Cx_newton = 2.0 * math.sin(Qk) ** 2
        gamma_gas = 1.4
        compr = 1.0 + 0.5 / (gamma_gas * M ** 2 * math.sin(Qk) ** 2)
        return min(Cx_newton * compr, 2.0)


# ─── Тепловой поток ───────────────────────────────────────────────────────────
def heat_flux(h_m, V):
    """
    Суммарный тепловой поток в точке торможения, Вт/м².

    q = q_turb + q_comp + q_amb

    q_turb — турбулентный конвективный нагрев при около-параболических скоростях
             (формула Воронцова, применима для CO2-атмосферы):
             q_turb = 1.15e6 * rho^0.8 / r_nose^0.2 * (V/Vc)^3.19
             где Vc = 7328 м/с — вторая космическая скорость Венеры,
             rho [кг/м³], r_nose [м].

    q_comp — нагрев за ударной волной (компрессионный/радиационный),
             значим при V > 8 км/с:
             q_comp = 7.845 * r_nose * (rho/rho0) * (V/1000)^8
             где rho0 = 64.79 кг/м³ — плотность у поверхности Венеры.

    q_amb  — конвективный нагрев от горячей атмосферы (важен ниже ~15 км,
             где T_atm > T_wall):
             q_amb = h_conv * max(0, T_atm - T_wall)
    """
    ro = Get_ro(h_m)

    # Турбулентный нагрев (Воронцов, при нулевом угле атаки)
    q_turb = 1.15e6 * (ro ** 0.8 / (r_nose ** 0.2)) * (V / 7328.0) ** 3.19

    # Компрессионный (радиационный) нагрев за ударной волной
    q_comp = (7.845 * r_nose) * (ro / 64.79) * (V / 1000.0) ** 8

    # Конвективный нагрев от атмосферы (нижние слои Венеры)
    q_amb = h_conv * max(0.0, T_atm(h_m) - T_wall)

    return max(0.0, q_turb + q_comp + q_amb)


# ─── Уравнения баллистического движения (без управления, без подъёмной силы) ─
def f_eom(V, th, R):
    """
    Правые части системы ОДУ баллистического движения в сферическом поле тяготения.

    Переменные состояния:
      V   — скорость [м/с]
      th  — угол наклона траектории к горизонту [рад], отрицателен при снижении
      R   — расстояние от центра планеты [м]

    Уравнения:
      dV/dt  = -D/m - g·sin(θ)
             D/m = 0.5·ρ·V²·Cx·S / m  (аэродинамическое торможение)

      dθ/dt  = cos(θ)·(V/R - g/V)
             (центробежный член V/R — гравитационный член g/V; подъёмная сила L=0)

      dR/dt  = V·sin(θ)
             (при θ<0 скорость снижения отрицательная — R убывает)

      dQ/dt  = q(h, V)
             (накопленный тепловой нагрев по траектории)
    """
    h = R - Rb
    ro = Get_ro(h)
    cx = Cx_func(V, h)
    gR = G0 * (Rb / R) ** 2          # гравитация на высоте R
    Dm = 0.5 * ro * V ** 2 * cx * S / mass   # аэродинамическое ускорение

    fV  = -gR * math.sin(th) - Dm                          # dV/dt
    fth = math.cos(th) * (V / R - gR / V)                  # dθ/dt  (исправлено: вынесен cos(th))
    fR  = V * math.sin(th)                                  # dR/dt
    fQ  = heat_flux(h, V)                                   # dQ/dt

    return fV, fth, fR, fQ


# ─── Интегрирование траектории (RK4) ──────────────────────────────────────────
def integrate_traj(h0, V0, theta0, save=False):
    """
    Интегрирует баллистическую траекторию от h0 до поверхности.
    Возвращает: V, theta, R, Q_total, t [, hist если save=True]
    """
    V  = float(V0)
    th = float(theta0)
    R  = Rb + float(h0)
    Q  = 0.0
    t  = 0.0

    if save:
        hist = {k: [] for k in ('t', 'V', 'theta_deg', 'h', 'Q', 'q')}

        def rec():
            hist['t'].append(t)
            hist['V'].append(V)
            hist['theta_deg'].append(th / DEG)
            hist['h'].append(R - Rb)
            hist['Q'].append(Q)
            hist['q'].append(heat_flux(R - Rb, V))
        rec()

    while R > Rb and t < T_MAX and V > 1.0:
        dt = DT_FINE if (R - Rb) < 3_000.0 else DT

        k1 = f_eom(V,                   th,                   R)
        k2 = f_eom(V + 0.5*dt*k1[0],  th + 0.5*dt*k1[1],  R + 0.5*dt*k1[2])
        k3 = f_eom(V + 0.5*dt*k2[0],  th + 0.5*dt*k2[1],  R + 0.5*dt*k2[2])
        k4 = f_eom(V +     dt*k3[0],  th +     dt*k3[1],  R +     dt*k3[2])
        c  = dt / 6.0

        V  += c * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        th += c * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        R  += c * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        Q  += c * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        t  += dt

        if R <= Rb:
            R = Rb

        if save:
            rec()

    if save:
        for k in hist:
            hist[k] = np.array(hist[k])
        return V, th, R, Q, t, hist
    return V, th, R, Q, t


# ─── Критерий оптимальности ───────────────────────────────────────────────────
def criterion(V_surface, Q_total):
    """
    Скалярный критерий: J = lambda_Q * Q - lambda_V * V → min
    Минимизация J одновременно:
      - минимизирует суммарный тепловой нагрев Q (Дж/м²)
      - максимизирует скорость удара о поверхность V (м/с)
    """
    return lambda_Q * Q_total - lambda_V * V_surface


def cost(x):
    """
    Целевая функция для оптимизатора.
    x = [V0, theta0]  — высота входа h0 ФИКСИРОВАНА (h0_ref).
    """
    V0, theta0 = x          # только 2 переменные
    try:
        V, _, R, Q, _ = integrate_traj(h0_ref, V0, theta0, save=False)
    except Exception:
        return 1e12
    pen = 0.0
    if R > Rb + 1.0:         # аппарат не достиг поверхности
        pen += 1e8 + 1e4 * (R - Rb)
    if V <= 0.0 or not np.isfinite(V):
        pen += 1e8
    return criterion(V, Q) + pen


# ─── Оптимизация начальных условий ───────────────────────────────────────────
def optimize_initial_conditions():
    sep = '═' * 60
    print(sep)
    print(' ОПТИМИЗАЦИЯ НАЧАЛЬНЫХ УСЛОВИЙ СПУСКА (БЕЗ УПРАВЛЕНИЯ)')
    print(f' Критерий: J = lambda_Q·Q(T) - lambda_V·V(T) → min')
    print(f' lambda_V={lambda_V:.3f}, lambda_Q={lambda_Q:.3e}')
    print(f' h0 = {h0_ref/1000:.0f} км (фиксирована)')
    print(f' V0   = [{BOUNDS[0][0]:.0f}, {BOUNDS[0][1]:.0f}] м/с')
    print(f' theta0 = [{BOUNDS[1][0]/DEG:.1f}, {BOUNDS[1][1]/DEG:.1f}]°')
    print(sep)

    Vb, _, _, Qb, _ = integrate_traj(h0_ref, V0_ref, theta0_ref)
    print(f' Базовый: V={Vb:.2f} м/с, Q={Qb/1e6:.4f} МДж/м², J={criterion(Vb, Qb):+.4e}')

    print('\nФаза 1 — дифференциальная эволюция (многопоточность) ...')
    t0 = time.time()
    _ctx = _mp_ctx()
    _pool = _ctx.Pool()
    res_de = differential_evolution(
        cost,
        BOUNDS,
        maxiter=120,
        popsize=18,
        tol=1e-7,
        mutation=(0.5, 1.2),
        recombination=0.9,
        seed=17,
        polish=False,
        disp=False,
        updating='deferred',
        workers=_pool.map,
    )
    _pool.close()
    _pool.join()
    print(f' {time.time()-t0:.1f}с  J={res_de.fun:.6e}  x={np.round(res_de.x, 4)}')

    print('Фаза 2 — Нелдер-Мид ...')
    t0 = time.time()
    res_nm = minimize(
        cost,
        res_de.x,
        method='Nelder-Mead',
        options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000, adaptive=True),
    )
    print(f' {time.time()-t0:.1f}с  J={res_nm.fun:.6e}  x={np.round(res_nm.x, 4)}')
    return res_nm


# ─── Параллельное интегрирование двух траекторий ─────────────────────────────
def _run_base(_):
    return integrate_traj(h0_ref, V0_ref, theta0_ref, save=True)


def _run_opti(x):
    V0, theta0 = x            # h0 фиксирована
    return integrate_traj(h0_ref, V0, theta0, save=True)


# ─── Вспомогательные функции результатов ─────────────────────────────────────
def _make_case(name, h0, V0, theta0, hist):
    q_max = float(np.max(hist['q']))
    V_f   = float(hist['V'][-1])
    Q_f   = float(hist['Q'][-1])
    T_f   = float(hist['t'][-1])
    return dict(name=name, h0=h0, V0=V0, theta0_deg=theta0 / DEG,
                Vf=V_f, Q=Q_f, q_max=q_max, T=T_f,
                J=criterion(V_f, Q_f), hist=hist)


def print_results(base, opti):
    sep = '─' * 60
    print(f'\n{sep}')
    print(f' РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ')
    print(sep)
    print(f'  {"":<34s} Базовый     Оптимум')
    print(f'  h0, км (фикс.)  {base["h0"]/1000:>12.3f} {opti["h0"]/1000:>10.3f}')
    print(f'  V0, м/с         {base["V0"]:>12.3f} {opti["V0"]:>10.3f}')
    print(f'  theta0, °       {base["theta0_deg"]:>12.3f} {opti["theta0_deg"]:>10.3f}')
    print(f'  T, с            {base["T"]:>12.3f} {opti["T"]:>10.3f}')
    print(f'  Vf, м/с         {base["Vf"]:>12.3f} {opti["Vf"]:>10.3f}')
    print(f'  Q, МДж/м²       {base["Q"]/1e6:>12.6f} {opti["Q"]/1e6:>10.6f}')
    print(f'  q_max, МВт/м²   {base["q_max"]/1e6:>12.6f} {opti["q_max"]/1e6:>10.6f}')
    print(f'  J               {base["J"]:>+12.4e} {opti["J"]:>+10.4e}')
    print(sep)
    print(f'  ΔVf    = {opti["Vf"]    - base["Vf"]:+.3f} м/с')
    print(f'  ΔQ     = {(opti["Q"]    - base["Q"]) / 1e6:+.6f} МДж/м²')
    print(f'  Δq_max = {(opti["q_max"]- base["q_max"]) / 1e6:+.6f} МВт/м²')
    print(sep)


def plot_results(base, opti):
    sup = (f'Оптимизация начальных условий спуска пенетратора (Венера)\n'
           f'h0={opti["h0"]/1000:.1f} км (фикс.), V0={opti["V0"]:.0f} м/с, '
           f'θ0={opti["theta0_deg"]:.2f}° | '
           f'Vf={opti["Vf"]:.1f} м/с, Q={opti["Q"]/1e6:.4f} МДж/м²')

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(sup, fontsize=10)
    b, o = base['hist'], opti['hist']

    ax = axs[0, 0]
    ax.plot(b['t'], b['h']/1000, 'k--', lw=1.8, label='Базовый')
    ax.plot(o['t'], o['h']/1000, 'b-',  lw=2.0, label='Оптимум')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Высота, км')
    ax.set_title('Высота'); ax.grid(True); ax.legend()

    ax = axs[0, 1]
    ax.plot(b['h']/1000, b['V']/1000, 'k--', lw=1.8, label='Базовый')
    ax.plot(o['h']/1000, o['V']/1000, 'b-',  lw=2.0, label='Оптимум')
    ax.set_xlabel('Высота, км'); ax.set_ylabel('Скорость, км/с')
    ax.set_title('Скорость'); ax.grid(True); ax.legend()

    ax = axs[1, 0]
    ax.semilogy(b['t'], np.maximum(b['q'], 1), 'k--', lw=1.8, label='Базовый')
    ax.semilogy(o['t'], np.maximum(o['q'], 1), 'r-',  lw=2.0, label='Оптимум')
    ax.set_xlabel('Время, с'); ax.set_ylabel('q, Вт/м²')
    ax.set_title('Тепловой поток'); ax.grid(True, which='both', alpha=0.4); ax.legend()

    ax = axs[1, 1]
    ax.plot(b['t'], b['Q']/1e6, 'k--', lw=1.8, label='Базовый')
    ax.plot(o['t'], o['Q']/1e6, 'g-',  lw=2.0, label='Оптимум')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Q, МДж/м²')
    ax.set_title('Накопленный нагрев'); ax.grid(True); ax.legend()

    plt.tight_layout()
    plt.savefig('output/penetrator_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _mp.freeze_support()
    t_start = time.time()

    res = optimize_initial_conditions()
    V0_opt, theta0_opt = res.x          # h0 ФИКСИРОВАНА — не распаковываем из res.x

    print('\nПараллельное интегрирование траекторий ...')
    ctx = _mp_ctx()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as exe:
        fut_base = exe.submit(_run_base, None)
        fut_opti = exe.submit(_run_opti, res.x)
        _, _, _, _, _, hist_b = fut_base.result()
        _, _, _, _, _, hist_o = fut_opti.result()

    base_case = _make_case('Базовый', h0_ref, V0_ref,  theta0_ref,   hist_b)
    opti_case = _make_case('Оптимум', h0_ref, V0_opt,  theta0_opt,   hist_o)

    print_results(base_case, opti_case)
    print(f'\nОбщее время: {time.time()-t_start:.1f} с')

    plot_results(base_case, opti_case)
