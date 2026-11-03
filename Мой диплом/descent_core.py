"""
descent_core.py — общее ядро расчёта спуска зонда-пенетратора в атмосфере Венеры
с учётом ветра (venus_wind.py). Используется обоими скриптами.

ПРОСТРАНСТВЕННАЯ (3D) МОДЕЛЬ ДВИЖЕНИЯ ЦЕНТРА МАСС
(невращающаяся сферическая планета, центральное поле тяготения, без подъёмной силы)
------------------------------------------------------------------------------------
Состояние: s = (V_E, V_N, V_U, φ, λ, r)
    V_E, V_N, V_U — восточная, северная и вертикальная составляющие путевой скорости
                    в местной системе ENU (восток–север–вверх);
    φ, λ          — широта и долгота; r — радиус.

    dV_E/dt = a_E + (V_E V_N tgφ - V_E V_U) / r
    dV_N/dt = a_N - (V_E² tgφ + V_N V_U) / r
    dV_U/dt = a_U - g(r) + (V_E² + V_N²) / r
    dφ/dt   = V_N / r
    dλ/dt   = V_E / (r cosφ)
    dr/dt   = V_U
Члены с 1/r — переносные (поворот местной системы при движении по сфере).

Это та же система, что и классическая запись через V, θ, ψ:
    dV/dt = a_τ - g sinθ
    dθ/dt = (a_n - g cosθ)/V + V cosθ / r
    dψ/dt = a_b / (V cosθ) + V cosθ sinψ tgφ / r
но без особенности при V cosθ -> 0 (почти вертикальное падение, разворот
горизонтальной скорости встречным ветром). V, θ, ψ вычисляются для вывода.

Ветер входит ТОЛЬКО через силу сопротивления, зависящую от скорости относительно воздуха:
    Va = V_путевая - W,   D = ½ ρ |Va|² Cx S,   a_D = -(D / m) · Va/|Va|.

Выходные координаты (как в исходной программе X — дальность, Y — высота, плюс Z):
    L — дальность вдоль начального большого круга (по азимуту входа ψ0),
    Z — боковое отклонение от него (вправо +), H = r - Rb.
Вращение Венеры не учитывается: ω = 2,99e-7 рад/с, кориолисово ускорение
2ωV ≤ 7e-3 м/с², окружная скорость поверхности ≤ 1,8 м/с.
"""

from __future__ import annotations

import math as m
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Константы Венеры
# ---------------------------------------------------------------------------
Rb = 6_051_800.0
g0 = 8.87
mass_planet = 4.867e24
gravy_const = 6.67e-11
cToDeg = 180.0 / m.pi
cToRad = m.pi / 180.0

# ---------------------------------------------------------------------------
# Таблицы атмосферы — ваши значения без изменений, вынесены из функций,
# чтобы не создавать списки заново на каждом шаге (результат тот же, быстрее).
# ИСПРАВЛЕНИЕ ПЛОТНОСТИ 102…118 км (FIX_RHO_100_120 = True):
# в исходной таблице этот блок — та же экспонента, что на 120…126 км (шкала высот
# 7,55 км), но умноженная ровно на 10: скачок ×13 между 120 и 118 км и рост
# плотности с высотой между 100 и 102 км — в гидростатической атмосфере так не бывает.
# Блок заменён логарифмически-линейной интерполяцией между ВАШИМИ опорными значениями
# на 100 км (1,347e-4) и 120 км (3,009e-6): шкала высот 5,26 км. Сверьте с VIRA;
# FIX_RHO_100_120 = False возвращает исходную таблицу RO_TAB_ORIG.
# ---------------------------------------------------------------------------
FIX_RHO_100_120 = True
H_TAB = [130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86,
         84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
         31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
         2, 1, 0]
RO_TAB_ORIG = [3.97200000e-08, 7.890e-7, 1.35931821e-06, 1.77164551e-06, 2.30904567e-06, 3.00945751e-06, 3.92232801e-05,
          5.11210308e-05, 6.66277727e-05, 8.68382352e-05, 1.13179216e-04, 1.47510310e-04, 1.92255188e-04, 2.50572705e-04,
          3.26579903e-04, 1.347e-4, 0.0002, 0.0004, 0.0007, 0.0012, 0.0019, 0.0031, 0.0049, 0.0077,
          0.0119, 0.0178, 0.0266, 0.0393, 0.0578, 0.0839, 0.1210, 0.1729, 0.2443, 0.3411, 0.4694, 0.6289,
          0.8183, 1.0320, 1.2840, 1.5940, 1.9670, 2.4260, 2.9850, 3.6460, 4.4040, 5.2760, 6.2740, 7.4200,
          8.7040, 9.4060, 10.1500, 10.9300, 11.7700, 12.6500, 13.5900, 14.5700, 15.6200, 16.7100, 17.8800,
          19.1100, 20.3900, 21.7400, 23.1800, 24.6800, 26.2700, 27.9500, 29.7400, 31.6000, 33.5400,
          35.5800, 37.7200, 39.9500, 42.2600, 44.7100, 47.2400, 49.8700, 52.6200, 55.4700, 58.4500, 61.5600,
          64.7900]


def _fix_rho(h_tab, ro_tab, h_lo=100, h_hi=120):
    ro = list(ro_tab)
    r_lo, r_hi = ro[h_tab.index(h_lo)], ro[h_tab.index(h_hi)]
    for i, hh in enumerate(h_tab):
        if h_lo < hh < h_hi:
            f = (hh - h_lo) / (h_hi - h_lo)
            ro[i] = r_lo * (r_hi / r_lo) ** f          # ln ρ линеен по высоте
    return ro


RO_TAB = _fix_rho(H_TAB, RO_TAB_ORIG) if FIX_RHO_100_120 else list(RO_TAB_ORIG)
A_TAB = [174, 176, 178, 180, 182, 185, 186, 187, 190, 193, 195, 196, 198, 199, 201, 203, 205, 206, 208.0, 208.0, 209.0,
         212.2, 215.4, 218.6, 221.8, 225.0, 228.2, 231.4, 234.6, 237.8, 241.0, 244.0, 247.0, 250.0, 253.0, 256.0, 263.2,
         270.4, 277.6, 284.8, 292.0, 296.8, 301.6, 306.4, 311.2, 316.0, 321.2, 326.4, 331.6, 336.8, 339.4, 342.0, 344.6,
         347.2, 349.8, 352.4, 355.0, 357.4, 359.8, 362.2, 364.6, 367.0, 369.4, 371.8, 374.2, 376.6, 379.0, 381.0, 383.0,
         385.0, 387.0, 389.0, 391.2, 393.4, 395.6, 397.8, 400.0, 402.0, 404.0, 406.0, 408.0, 410.0]


# --- ваша интерполяция Ньютона (без изменений по смыслу) --------------------
def find_closest_points_ro(x, xi):
    closest_points = []
    for i in range(len(x)):
        if xi - 2 <= x[i] <= xi + 2:
            if i <= 2:
                closest_indices = list(range(4))
            elif i >= len(x) - 2:
                closest_indices = list(range(len(x) - 4, len(x)))
            else:
                closest_indices = list(range(i - 2, i + 2))
            closest_points = [x[idx] for idx in closest_indices]
            break
    return closest_points


def divided_diff(x, y):
    n = len(y)
    coef = [0] * n
    coef[0] = y[0]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            if x[i] == x[i - j]:
                coef[i] = y[i]
            else:
                y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])
                coef[i] = y[i]
    return coef


def newton_interpolation_ro(x, y, xi):
    xp = find_closest_points_ro(x, xi)
    yp = [y[x.index(v)] for v in xp]
    coef = divided_diff(xp, yp)
    n = len(coef) - 1
    result = coef[n]
    for i in range(n - 1, -1, -1):
        result = result * (xi - xp[i]) + coef[i]
    return result


def find_closest_points(x, xi):
    for i in range(len(x)):
        if x[i] >= xi:
            return [x[i], x[i + 1]] if i == 0 else [x[i - 1], x[i]]
    return []


def newton_interpolation(x, y, xi):
    xp = find_closest_points(x, xi)
    yp = [y[x.index(v)] for v in xp]
    coef = divided_diff(xp, yp)
    n = len(coef) - 1
    result = coef[n]
    for i in range(n - 1, -1, -1):
        result = result * (xi - xp[i]) + coef[i]
    return result


def Get_ro(h_m):
    return newton_interpolation_ro(H_TAB, RO_TAB, h_m / 1000.0)


def v_sound(h_m):
    return newton_interpolation_ro(H_TAB, A_TAB, h_m / 1000.0)


# ---------------------------------------------------------------------------
# Правая часть и интегратор (3D)
# ---------------------------------------------------------------------------
def rhs(s, ro, Cxa, S, mass, wind):
    """s = (VE, VN, VU, phi, lam, r). ro и Cxa фиксируются на шаге (как в исходной программе)."""
    VE, VN, VU, phi, lam, r = s
    if wind is not None:
        u, v, w = wind(r - Rb)
    else:
        u = v = w = 0.0

    # скорость относительно воздуха
    aE, aN, aU = VE - u, VN - v, VU - w
    Va = m.sqrt(aE * aE + aN * aN + aU * aU)
    k = 0.5 * ro * Va * Cxa * S / mass          # |D| / (m·Va)

    gR = g0 * Rb * Rb / (r * r)
    tphi = m.tan(phi)
    dVE = -k * aE + (VE * VN * tphi - VE * VU) / r
    dVN = -k * aN - (VE * VE * tphi + VN * VU) / r
    dVU = -k * aU - gR + (VE * VE + VN * VN) / r
    return (dVE, dVN, dVU, VN / r, VE / (r * m.cos(phi)), VU)


def rk4_step(s, dt, *args):
    k1 = rhs(s, *args)
    s2 = tuple(a + 0.5 * dt * b for a, b in zip(s, k1))
    k2 = rhs(s2, *args)
    s3 = tuple(a + 0.5 * dt * b for a, b in zip(s, k2))
    k3 = rhs(s3, *args)
    s4 = tuple(a + dt * b for a, b in zip(s, k3))
    k4 = rhs(s4, *args)
    return tuple(a + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
                 for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4))


def flight_angles(VE, VN, VU):
    """Модуль путевой скорости V, траекторный угол θ, путевой азимут ψ (от севера по часовой)."""
    V = m.sqrt(VE * VE + VN * VN + VU * VU)
    return V, m.asin(VU / V), m.atan2(VE, VN) % (2 * m.pi)


def wind_enu(wind, h):
    return wind(h) if wind is not None else (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Пересчёт широты/долготы в дальность L и боковое отклонение Z
# относительно большого круга, проходящего через точку входа по азимуту ψ0
# ---------------------------------------------------------------------------
def track_frame(phi0, lam0, psi0):
    n0 = np.array([m.cos(phi0) * m.cos(lam0), m.cos(phi0) * m.sin(lam0), m.sin(phi0)])
    eE = np.array([-m.sin(lam0), m.cos(lam0), 0.0])
    eN = np.array([-m.sin(phi0) * m.cos(lam0), -m.sin(phi0) * m.sin(lam0), m.cos(phi0)])
    t = m.sin(psi0) * eE + m.cos(psi0) * eN       # направление трассы
    b = m.cos(psi0) * eE - m.sin(psi0) * eN       # вправо от трассы
    return n0, t, b


def downrange_crossrange(phi, lam, frame):
    n0, t, b = frame
    phi, lam = np.asarray(phi), np.asarray(lam)
    p = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
    L = Rb * np.arctan2(p @ t, p @ n0)
    Z = Rb * np.arcsin(np.clip(p @ b, -1.0, 1.0))
    return L, Z


@dataclass
class DescentParams:
    mass: float
    S: float
    V0: float
    tetta0: float           # рад, траекторный угол входа (< 0)
    azimuth: float          # рад, азимут входа ψ0 от севера по часовой
    lat0: float = 0.0       # рад, широта точки входа
    lon0: float = 0.0       # рад, долгота точки входа
    h0: float = 125_000.0
    dt: float = 0.01
    save_every: int = 10    # сохранять каждый N-й шаг (экономит память в пуле процессов)


def simulate(p: DescentParams, cx_func, wind=None, tag=None):
    """
    cx_func(Mach, h_m) -> Cxa. Возвращает словарь numpy-массивов + итоговые величины.
    Mach и скоростной напор считаются по ВОЗДУШНОЙ скорости.
    """
    ct0 = m.cos(p.tetta0)
    s = (p.V0 * ct0 * m.sin(p.azimuth), p.V0 * ct0 * m.cos(p.azimuth), p.V0 * m.sin(p.tetta0),
         p.lat0, p.lon0, Rb + p.h0)
    t = 0.0
    step = 0
    keys = ("T", "H", "lat", "lon", "V", "Va", "TETTA", "PSI", "napor", "nx", "PX", "CX", "MAH",
            "Wu", "Wv", "Ww", "Wx", "Wz")
    rec = {k: [] for k in keys}

    while s[5] >= Rb:
        VE, VN, VU, phi, lam, r = s
        h = r - Rb
        u, v, w = wind_enu(wind, h)
        Va = m.sqrt((VE - u) ** 2 + (VN - v) ** 2 + (VU - w) ** 2)
        a_snd = v_sound(h)
        ro = Get_ro(h)
        Mach = Va / a_snd
        Cxa = cx_func(Mach, h)

        if step % p.save_every == 0:
            V, th, psi = flight_angles(VE, VN, VU)
            gR = gravy_const * mass_planet / r ** 2
            q = 0.5 * ro * Va * Va
            Wx = u * m.sin(psi) + v * m.cos(psi)     # ветер вдоль текущего путевого азимута (+ попутный)
            Wz = u * m.cos(psi) - v * m.sin(psi)     # поперёк (+ вправо)
            for k_, v_ in zip(keys, (t, h, phi, lam, V, Va, th * cToDeg, psi * cToDeg, q,
                                     q * Cxa * p.S / (p.mass * gR), p.mass / (Cxa * p.S),
                                     Cxa, Mach, u, v, w, Wx, Wz)):
                rec[k_].append(v_)

        s_prev = s
        s = rk4_step(s, p.dt, ro, Cxa, p.S, p.mass, wind)
        t += p.dt
        step += 1

    # точка падения: линейная интерполяция на h = 0 внутри последнего шага
    f = (s_prev[5] - Rb) / (s_prev[5] - s[5])
    sl = tuple(a + f * (b - a) for a, b in zip(s_prev, s))
    frame = track_frame(p.lat0, p.lon0, p.azimuth)
    Lland, Zland = downrange_crossrange(sl[3], sl[4], frame)
    Vl, thl, psil = flight_angles(sl[0], sl[1], sl[2])
    land = dict(V=Vl, tetta=thl, psi=psil, L=float(Lland), Z=float(Zland),
                lat=sl[3], lon=sl[4], t=t - p.dt + f * p.dt)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["L"], out["Z"] = downrange_crossrange(out["lat"], out["lon"], frame)
    out["acceleration"] = np.gradient(out["V"], out["T"]) if out["T"].size > 1 else np.zeros(1)
    out["landing"] = land
    out["tag"] = tag
    out["V0"], out["tetta0"], out["azimuth"] = p.V0, p.tetta0, p.azimuth
    if wind is not None:   # профиль ветра реализации (прорежен) — для графиков
        out["wind_h"] = wind.h[::20]
        out["wind_u"] = wind.u[::20]
        out["wind_v"] = wind.v[::20]
    return out

# ---------------------------------------------------------------------------
# Графики в вашем оформлении
# ---------------------------------------------------------------------------
FONT = dict(fontsize=16, fontname="Times New Roman")


def plot_family(results, xkey, ykey, title, xlabel, ylabel, left=0.15, show=True, savepath=None):
    import matplotlib.pyplot as plt
    plt.figure()
    for i, r in enumerate(results):
        plt.plot(r[xkey], r[ykey], label=f"Вариант {i + 1}", lw=0.8)
    plt.title(title, **FONT)
    plt.xlabel(xlabel, **FONT)
    plt.ylabel(ylabel, **FONT)
    plt.subplots_adjust(left=left, right=0.95, top=0.9, bottom=0.15)
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()


def plot_standard_set(results, show=True, prefix=None):
    """Тот же набор графиков, что был в исходных программах, + ветер."""
    sp = (lambda n: f"{prefix}_{n}.png") if prefix else (lambda n: None)
    plot_family(results, "L", "H", "Траектории спуска зонда-пенетратора", "Дальность, м", "Высота, м", show=show, savepath=sp("traj"))
    plot_family(results, "T", "H", "Зависимость высоты от времени", "Время, с", "Высота, м", show=show, savepath=sp("h_t"))
    plot_family(results, "T", "V", "Зависимость модуля скорости от времени", "Время, c", r"Скорость, $\frac{м}{с}$", show=show, savepath=sp("v_t"))
    plot_family(results, "H", "V", "Зависимость модуля скорости от высоты", "Высота, м", r"Скорость, $\frac{м}{с}$", show=show, savepath=sp("v_h"))
    plot_family(results, "T", "TETTA", "Зависимость траекторного угла от времени", "Время, c", "Траекторный угол, град", show=show, savepath=sp("tetta_t"))
    plot_family(results, "T", "napor", "Зависимость скоростного напора от времени", "Время, с",
                r"Скоростной напор, $\frac{\mathrm{кг}}{\mathrm{м} \cdot \mathrm{с}^{2}}$", left=0.25, show=show, savepath=sp("q_t"))
    plot_family(results, "T", "acceleration", "Зависимость ускорения от времени", "Время, с", r"Ускорение, м/с$^2$", show=show, savepath=sp("acc_t"))
    plot_family(results, "T", "nx", "Зависимость перегрузки от времени", "Время, с", "Перегрузка, g", show=show, savepath=sp("nx_t"))
    plot_family(results, "T", "PX", "Зависимость давления на мидель от времени", "Время, с", r"Px, $\frac{кг}{м^2}$", show=show, savepath=sp("px_t"))
    plot_family(results, "MAH", "CX", "Зависимость CX от М", "Мах (по воздушной скорости)", "Cx", show=show, savepath=sp("cx_m"))
    plot_family(results, "Wx", "H", "Продольный ветер вдоль траектории", "Wx, м/с (+ попутный)", "Высота, м", show=show, savepath=sp("wx_h"))
    plot_family(results, "Wz", "H", "Боковой ветер вдоль траектории", "Wz, м/с (+ вправо)", "Высота, м", show=show, savepath=sp("wz_h"))
    plot_family(results, "Z", "H", "Боковой снос", "Снос Z, м", "Высота, м", show=show, savepath=sp("z_h"))
    plot_family(results, "L", "Z", "Проекция траекторий на поверхность", "Дальность L, м", "Боковое отклонение Z, м", show=show, savepath=sp("plan"))
    plot_family(results, "T", "PSI", "Путевой азимут от времени", "Время, с", "ψ, град", show=show, savepath=sp("psi_t"))
    plot_3d(results, show=show, savepath=sp("traj3d"))


def plot_3d(results, show=True, savepath=None):
    """Пространственные траектории: X = L (дальность), Y = Z (боковое), Z-ось графика = высота."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    for r in results:
        ax.plot(r["L"] / 1000, r["Z"] / 1000, r["H"] / 1000, lw=0.8)
    ax.set_xlabel("Дальность L, км")
    ax.set_ylabel("Боковое Z, км")
    ax.set_zlabel("Высота H, км")
    ax.set_title("Пространственные траектории спуска", **FONT)
    if savepath:
        fig.savefig(savepath, dpi=130)
    if show:
        plt.show()
    else:
        plt.close(fig)


def landing_stats(results):
    L = np.array([r["landing"]["L"] for r in results])
    Z = np.array([r["landing"]["Z"] for r in results])
    V = np.array([r["landing"]["V"] for r in results])
    t = np.array([r["landing"]["t"] for r in results])
    nx = np.array([r["nx"].max() for r in results])
    q = np.array([r["napor"].max() for r in results])
    return dict(L=L, Z=Z, V=V, t=t, nx_max=nx, q_max=q)


def plot_landing(results, show=True, savepath=None):
    """Точки падения и эллипс рассеивания 3σ."""
    import matplotlib.pyplot as plt
    st = landing_stats(results)
    L, Z = st["L"] / 1000.0, st["Z"] / 1000.0
    plt.figure()
    plt.plot(L, Z, "o", ms=4)
    if L.size >= 3:
        C = np.cov(np.vstack([L, Z]))
        w, vec = np.linalg.eigh(C)
        ang = np.linspace(0, 2 * np.pi, 200)
        ell = vec @ (3 * np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
        plt.plot(L.mean() + ell[0], Z.mean() + ell[1], "r", label="эллипс 3σ")
        plt.legend()
    plt.title("Рассеивание точек падения", **FONT)
    plt.xlabel("Дальность L, км", **FONT)
    plt.ylabel("Боковой снос Z, км", **FONT)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=130)
    if show:
        plt.show()
    else:
        plt.close()
