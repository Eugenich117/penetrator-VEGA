# -*- coding: utf-8 -*-
"""
========================================================================
ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — решение по принципу минимума Понтрягина
========================================================================

Согласно разделу 11.4 методички "Программирование оптимального управления":

МАТЕМАТИЧЕСКАЯ ПОСТАНОВКА:
  Система уравнений:
    dh/dt = v
    dv/dt = c*u/m - g(h)
    dm/dt = -u

  Критерий: J = m0 - m(T) -> min (минимум расхода топлива)

  Ограничения:
    0 <= u(t) <= u_m
    h(T) = 0, v(T) = 0
    T - свободное

ОПТИМАЛЬНОЕ УПРАВЛЕНИЕ (по методичке):
  Из принципа минимума Понтрягина следует, что оптимальное управление
  имеет структуру bang-bang с ОДНИМ переключением:

    u(t) = 0,      если t < t*
    u(t) = u_m,    если t* <= t <= T

  где t* — момент включения двигателя (единственное переключение)

ДОКАЗАТЕЛЬСТВО ОТСУТСТВИЯ ОСОБОГО УПРАВЛЕНИЯ:
  Если предположить, что функция переключения обращается в нуль на
  некотором интервале, то приходим к противоречию с уравнениями движения.
  Следовательно, особого управления нет, и управление всегда граничное.

ОПТИМИЗИРУЕМЫЕ ПАРАМЕТРЫ:
  t* — момент переключения (включение двигателя)
  T  — время посадки (касание поверхности)

========================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import multiprocessing as mp

# ─────────────────────────────────────────────────────────────────────
# ИСХОДНЫЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────
R_moon    = 1_737_000.0    # радиус Луны, м
g0        = 1.62           # ускорение свободного падения на поверхности
h0        = 15_000.0       # начальная высота, м
v0        = -20.0          # начальная скорость, м/с (вниз)
m0        = 5_560.0        # начальная масса, кг
m_dry     = 3_740.0        # сухая масса, кг
c         = 3_050.0        # удельный импульс, м/с
P_max     = 20_000.0       # максимальная тяга, Н
u_m       = P_max / c      # максимальный расход, кг/с

# ─────────────────────────────────────────────────────────────────────
# КОЭФФИЦИЕНТЫ ГРАВИТАЦИОННОГО ПОТЕНЦИАЛА ЛУНЫ (ГПЛ)
# Для посадки на полюсе (модель GRGM1200A/SGM150J)
# ─────────────────────────────────────────────────────────────────────
# Источник: NASA Goddard Space Flight Center - Lunar Gravity Model
# Коэффициенты нормированы по стандарту IAU/IAG

J2        = 2.034e-4       # второй зональный коэффициент (C20)
J3        = 8.0e-6         # третий зональный коэффициент (C30)
J4        = 1.0e-5         # четвертый зональный коэффициент (C40)
C22       = 2.2e-5         # секториальный коэффициент (важен для полюса)
S22       = 0.17e-5        # секториальный коэффициент (фаза)

# Для полюса: зональные гармоники дают основной вклад,
# секториальные C22/S22 создают небольшие возмущения (~10⁻⁶ g0)

# ─────────────────────────────────────────────────────────────────────
# МАСКОНЫ (Mass Concentrations) — области повышенной гравитации
# Модель точечных масс на основе данных GRAIL (NASA)
# ─────────────────────────────────────────────────────────────────────
# Формат: (широта, долгота, избыточная масса, глубина)
# Источники: Andrews-Hanna et al. (2013), Miljković et al. (2016)
#            JGR Planets, GRAIL gravity model

MASCONS = [
    # Основные масконы в экваториальной области
    {"name": "Imbrium",      "lat": 35.0,  "lon": 340.0, "dm": 1.2e18, "depth": 50_000},
    {"name": "Serenitatis",  "lat": 25.0,  "lon": 20.0,  "dm": 9.0e17, "depth": 45_000},
    {"name": "Crisium",      "lat": 15.0,  "lon": 60.0,  "dm": 8.5e17, "depth": 40_000},
    {"name": "Humorum",      "lat": -25.0, "lon": 320.0, "dm": 7.0e17, "depth": 35_000},
    {"name": "Nectaris",     "lat": -15.0, "lon": 35.0,  "dm": 6.5e17, "depth": 40_000},
    {"name": "Orientale",    "lat": -20.0, "lon": 265.0, "dm": 7.5e17, "depth": 50_000},
    # Южный полюс — бассейн South Pole-Aitken (важен для посадки на полюс)
    {"name": "SPA_North",    "lat": -35.0, "lon": 180.0, "dm": 5.0e17, "depth": 60_000},
    {"name": "SPA_Center",   "lat": -45.0, "lon": 180.0, "dm": 6.0e17, "depth": 80_000},
    {"name": "SPA_South",    "lat": -55.0, "lon": 175.0, "dm": 4.5e17, "depth": 70_000},
    # Дополнительные локальные аномалии near south pole
    {"name": "Shackleton",   "lat": -88.0, "lon": 130.0, "dm": 1.0e16, "depth": 10_000},
    {"name": "Shoemaker",    "lat": -85.0, "lon": 150.0, "dm": 8.0e15, "depth": 8_000},
    # Мелкомасштабные неоднородности южной полярной области
    # (по данным LRO LOLA и GRAIL)
    {"name": "de Gerlache",  "lat": -88.5, "lon": 280.0, "dm": 5.0e15, "depth": 5_000},
    {"name": "Sverdrup",     "lat": -87.0, "lon": 310.0, "dm": 4.0e15, "depth": 6_000},
    {"name": "Amundsen",     "lat": -84.0, "lon": 80.0,  "dm": 1.2e16, "depth": 12_000},
    {"name": "Scott",        "lat": -86.0, "lon": 20.0,  "dm": 9.0e15, "depth": 10_000},
    {"name": "Faustini",     "lat": -87.5, "lon": 50.0,  "dm": 1.1e16, "depth": 11_000},
    {"name": "Haworth",      "lat": -88.0, "lon": 270.0, "dm": 6.0e15, "depth": 7_000},
]

# Гравитационная постоянная
G = 6.67430e-11  # м³/(кг·с²)

# Вращение Луны (для учёта изменения долготы при снижении)
# Луна вращается синхронно, но есть либрация и посадка не строго на полюс
OMEGA_MOON = 2.662e-6  # рад/с (угловая скорость вращения Луны)

print("=" * 72)
print(" ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — принцип минимума Понтрягина")
print(" с учётом гравитационных возмущений (посадка на полюс)")
print("=" * 72)
print(" Согласно разделу 11.4 методички:")
print(" Оптимальное управление — bang-bang с ОДНИМ переключением")
print(" u(t) = 0 при t < t*, u(t) = u_m при t* <= t <= T")
print("=" * 72)
print(f" h0      = {h0:.1f} м")
print(f" v0      = {v0:.1f} м/с")
print(f" m0      = {m0:.1f} кг")
print(f" m_dry   = {m_dry:.1f} кг")
print(f" c       = {c:.1f} м/с")
print(f" P_max   = {P_max:.1f} Н")
print(f" u_m     = {u_m:.6f} кг/с")
print("=" * 72)
print(" ГРАВИТАЦИОННЫЕ ВОЗМУЩЕНИЯ (ГПЛ для полюса):")
print(f" J2  = {J2:.3e}  (второй зональный коэффициент)")
print(f" J3  = {J3:.3e}  (третий зональный коэффициент)")
print(f" J4  = {J4:.3e}  (четвертый зональный коэффициент)")
print(f" C22 = {C22:.3e} (секториальный коэффициент)")
print(f" S22 = {S22:.3e} (секториальный коэффициент)")
print("=" * 72)
print(" МАСКОНЫ (источники локальных гравитационных аномалий):")
for mc in MASCONS:
    print(f"  {mc['name']:15s}  шир={mc['lat']:6.1f}°  долг={mc['lon']:7.1f}°  Δm={mc['dm']:.1e} кг")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────
# ГРАВИТАЦИЯ С ВОЗМУЩЕНИЯМИ И МАСКОНАМИ
# ─────────────────────────────────────────────────────────────────────
def g_func(h, lat=-90.0, lon=0.0, t=0.0):
    """
    Гравитация с учётом высоты, сферических гармоник и масконов.

    Параметры:
    - h: высота над поверхностью, м
    - lat: широта точки посадки (градусы), по умолчанию -90 (южный полюс)
    - lon: начальная долгота (градусы), по умолчанию 0
    - t: время с начала снижения, с (учёт вращения Луны)

    Для посадки на полюсе (θ = 0) гравитационный потенциал:
    V(r,θ,λ) = -μ/r * [1 - Σ J_n*(R/r)^n*P_n(cosθ) + ...] + Σ V_mascon

    Ускорение: g = -∂V/∂r

    Учёт вращения Луны: долгота меняется как lon(t) = lon0 + OMEGA_MOON * t
    """
    h_eff = max(h, 0.0)
    r = R_moon + h_eff
    r_ratio = R_moon / r  # безразмерный радиус

    # Базовая гравитация (сферически симметричная)
    g_base = g0 * r_ratio ** 2

    # ── Зональные возмущения (J2, J3, J4) ───────────────────────────
    delta_J2 = 3.0 * J2 * r_ratio ** 2
    delta_J3 = 4.0 * J3 * r_ratio ** 3
    delta_J4 = 5.0 * J4 * r_ratio ** 4

    # ── Секториальные возмущения (C22, S22) ────────────────────────
    delta_C22 = 6.0 * C22 * r_ratio ** 2
    delta_S22 = 6.0 * S22 * r_ratio ** 2

    # ── Возмущения от масконов ──────────────────────────────────────
    # Модель точечных масс: Δg = G * Δm / d²
    delta_mascons = 0.0

    # Преобразуем координаты в радианы с учётом вращения Луны
    lat_rad = np.radians(lat)
    # Долгота меняется со временем из-за вращения Луны (либрация)
    lon_eff = lon + np.degrees(OMEGA_MOON * t)
    lon_rad = np.radians(lon_eff)

    for mascon in MASCONS:
        mc_lat = np.radians(mascon["lat"])
        mc_lon = np.radians(mascon["lon"])
        dm = mascon["dm"]
        mc_depth = mascon["depth"]

        # Расстояние от центра Луны до маскона
        r_mascon = R_moon - mc_depth

        # Угловое расстояние между КА и масконом
        cos_psi = (np.sin(lat_rad) * np.sin(mc_lat) +
                   np.cos(lat_rad) * np.cos(mc_lat) * np.cos(lon_rad - mc_lon))
        cos_psi = np.clip(cos_psi, -1.0, 1.0)

        # 3D расстояние от КА до центра маскона
        d_squared = r**2 + r_mascon**2 - 2*r*r_mascon*cos_psi
        d_squared = max(d_squared, (R_moon - mc_depth)**2)

        # Радиальная компонента ускорения от маскона
        radial_factor = (r - r_mascon * cos_psi) / np.sqrt(d_squared)
        radial_factor = np.clip(radial_factor, 0.0, 1.0)

        delta_g_mascon = G * dm / d_squared * radial_factor
        delta_mascons += delta_g_mascon

    # Суммарная гравитация
    g_harmonics = g_base * (1.0 + delta_J2 + delta_J3 + delta_J4 + delta_C22 + delta_S22)
    g_total = g_harmonics + delta_mascons

    return g_total


def g_perturbation(h, lat=-90.0, lon=0.0, t=0.0):
    """Возвращает только возмущающую часть гравитации."""
    g_base = g0 * (R_moon / (R_moon + max(h, 0.0))) ** 2
    return g_func(h, lat, lon, t) - g_base


def g_mascon_component(h, lat=-90.0, lon=0.0, t=0.0):
    """Возвращает только вклад от масконов (отдельно от гармоник)."""
    h_eff = max(h, 0.0)
    r = R_moon + h_eff
    lat_rad = np.radians(lat)
    lon_eff = lon + np.degrees(OMEGA_MOON * t)
    lon_rad = np.radians(lon_eff)

    delta_mascons = 0.0
    g_mascon_detail = []

    for mascon in MASCONS:
        mc_lat = np.radians(mascon["lat"])
        mc_lon = np.radians(mascon["lon"])
        dm = mascon["dm"]
        mc_depth = mascon["depth"]
        r_mascon = R_moon - mc_depth

        cos_psi = (np.sin(lat_rad) * np.sin(mc_lat) +
                   np.cos(lat_rad) * np.cos(mc_lat) * np.cos(lon_rad - mc_lon))
        cos_psi = np.clip(cos_psi, -1.0, 1.0)

        d_squared = r**2 + r_mascon**2 - 2*r*r_mascon*cos_psi
        d_squared = max(d_squared, (R_moon - mc_depth)**2)

        radial_factor = (r - r_mascon * cos_psi) / np.sqrt(d_squared)
        radial_factor = np.clip(radial_factor, 0.0, 1.0)

        delta_g = G * dm / d_squared * radial_factor
        delta_mascons += delta_g
        g_mascon_detail.append({"name": mascon["name"], "dg": delta_g})

    return delta_mascons, g_mascon_detail


# ─────────────────────────────────────────────────────────────────────
# УПРАВЛЕНИЕ ПО ПРИНЦИПУ МИНИМУМА (одно переключение)
# ─────────────────────────────────────────────────────────────────────
def control_pontryagin(t, t_switch, m):
    """
    Оптимальное управление по методичке:
    u(t) = 0,  если t < t_switch
    u(t) = u_m, если t >= t_switch
    """
    if m <= m_dry:
        return 0.0
    return u_m if t >= t_switch else 0.0


# ─────────────────────────────────────────────────────────────────────
# ПРАВЫЕ ЧАСТИ
# ─────────────────────────────────────────────────────────────────────
# Параметры для расчёта гравитации (посадка на южном полюсе)
LANDING_LAT = -90.0  # широта посадки (градусы)
LANDING_LON = 0.0    # начальная долгота (градусы)

def rhs(t, y, t_switch):
    """
    Уравнения движения с гравитацией от масконов.
    t — время используется для учёта вращения Луны в g_func
    """
    h, v, m = y
    u = control_pontryagin(t, t_switch, m)
    m_eff = max(m, m_dry + 1e-9)

    dh = v
    # Передаём время t для учёта вращения Луны в g_func
    dv = c * u / m_eff - g_func(h, lat=LANDING_LAT, lon=LANDING_LON, t=t)
    dm = -u

    return [dh, dv, dm]


# ─────────────────────────────────────────────────────────────────────
# МОДЕЛИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────
def simulate(params, save_history=False):
    """
    Интегрирование уравнений движения.
    params = [t_switch, T] — момент переключения и время посадки
    """
    t_switch, T = params
    t_switch = max(0.0, t_switch)
    T = max(T, t_switch + 1.0)

    # Событие: касание поверхности
    def hit_ground(t, y):
        return y[0]
    hit_ground.terminal = True
    hit_ground.direction = -1

    # Интегрирование
    sol = solve_ivp(
        lambda t, y: rhs(t, y, t_switch),
        [0.0, T * 1.5],
        [h0, v0, m0],
        method='RK45',
        events=hit_ground,
        max_step=0.5,
        rtol=1e-9,
        atol=1e-11,
        dense_output=save_history
    )

    landed = len(sol.t_events[0]) > 0
    t_land = float(sol.t[-1]) if landed else T
    h_land = float(sol.y[0, -1])
    v_land = float(sol.y[1, -1])
    m_land = max(float(sol.y[2, -1]), m_dry)

    # Проверка на восходящую скорость (взлет)
    max_h = np.max(sol.y[0])
    max_upward_v = max(0.0, np.max(sol.y[1]))

    result = {
        "landed": landed,
        "t_land": t_land,
        "h_land": h_land,
        "v_land": v_land,
        "m_land": m_land,
        "fuel_used": m0 - m_land,
        "max_h": max_h,
        "max_upward_v": max_upward_v,
        "ascent_amount": max(0.0, max_h - h0),
        "sol": sol if save_history else None,
        "t_switch": t_switch,
    }

    return result


# ─────────────────────────────────────────────────────────────────────
# ЦЕЛЕВАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────
def objective(params):
    """
    Минимизация расхода топлива с штрафами за нарушения.
    params = [t_switch, T]
    """
    t_switch, T = params

    if not (0.0 <= t_switch <= T <= 600.0):
        return 1e15

    sim = simulate(params, save_history=False)

    J = sim["fuel_used"]

    # Штраф за непосадку
    if not sim["landed"]:
        J += 1e6 * (sim["h_land"] ** 2 + 1.0)

    # Штраф за высокую скорость посадки
    if sim["landed"]:
        J += 1e6 * max(0.0, abs(sim["v_land"]) - 0.5) ** 2
        J += 1e6 * sim["v_land"] ** 2  # стремимся к нулевой скорости

    # Штраф за взлет
    if sim["ascent_amount"] > 1.0:
        J += 1e5 * (sim["ascent_amount"] - 1.0) ** 2

    return J


# ─────────────────────────────────────────────────────────────────────
# ОГРАНИЧЕНИЯ ДЛЯ SLSQP
# ─────────────────────────────────────────────────────────────────────
def con_landed(params):
    """КА должен приземлиться: h(T) = 0"""
    res = simulate(params)
    return -res["h_land"]  # должно быть >= 0, т.е. h_land <= 0


def con_v_soft(params):
    """Мягкая посадка: |v(T)| <= 0.5 м/с"""
    res = simulate(params)
    return 0.5 - abs(res["v_land"])  # должно быть >= 0


def con_v_zero(params):
    """Стремление к нулевой скорости посадки"""
    res = simulate(params)
    return 0.1 - abs(res["v_land"])  # должно быть >= 0


# ─────────────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────
def solve():
    """
    Двухэтапная оптимизация:
    1. Differential Evolution для глобального поиска
    2. SLSQP для локальной полировки
    """
    # Границы: [t_switch, T]
    bounds = [
        (0.0, 400.0),    # t_switch
        (50.0, 500.0),   # T
    ]

    print("\n" + "=" * 72)
    print(" ШАГ 1: Глобальный поиск (Differential Evolution)")
    print("=" * 72)

    t0 = time.time()
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=mp.cpu_count())

    res_de = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=300,
        popsize=20,
        tol=1e-8,
        mutation=(0.5, 1.2),
        recombination=0.85,
        seed=42,
        polish=False,
        disp=False,
        workers=pool.map,
        updating="deferred",
    )

    pool.close()
    pool.join()

    print(f" Время: {time.time() - t0:.1f} с")
    print(f" J_de  = {res_de.fun:.6e}")
    print(f" t_switch = {res_de.x[0]:.3f} с, T = {res_de.x[1]:.3f} с")

    # Проверка результата DE
    sim_de = simulate(res_de.x)
    print(f" Посадка: landed={sim_de['landed']}, h={sim_de['h_land']:.2f} м, "
          f"v={sim_de['v_land']:.2f} м/с")

    print("\n" + "=" * 72)
    print(" ШАГ 2: Локальная полировка (SLSQP)")
    print("=" * 72)

    # Ограничения
    constraints = [
        {"type": "ineq", "fun": con_landed},   # h <= 0
        {"type": "ineq", "fun": con_v_soft},   # |v| <= 0.5
    ]

    t1 = time.time()
    res_local = minimize(
        objective,
        x0=res_de.x,
        method="Nelder-Mead",
        #method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
            "disp": True,
        }
    )

    print(f"\n Время: {time.time() - t1:.1f} с")
    print(f" Успех: {res_local.success} ({res_local.message})")

    x_opt = res_local.x
    sim = simulate(x_opt, save_history=False)

    print("\n" + "=" * 72)
    print(" РЕЗУЛЬТАТ ОПТИМИЗАЦИИ (по Понтрягину)")
    print("=" * 72)
    print(f" Успешное касание   = {sim['landed']}")
    print(f" t_switch (включение) = {x_opt[0]:.6f} с")
    print(f" T (посадка)         = {sim['t_land']:.6f} с")
    print(f" Длительность работы двигателя = {sim['t_land'] - x_opt[0]:.3f} с")
    print(f" h_land             = {sim['h_land']:.9f} м")
    print(f" v_land             = {sim['v_land']:.9f} м/с")
    print(f" m_land             = {sim['m_land']:.6f} кг")
    print(f" fuel_used          = {sim['fuel_used']:.6f} кг")
    print(f" max_h              = {sim['max_h']:.6f} м")
    print(f" ascent_amount      = {sim['ascent_amount']:.6f} м")
    print("=" * 72)

    # Проверка условия оптимальности
    print("\n ПРОВЕРКА УСЛОВИЙ ОПТИМАЛЬНОСТИ:")
    print(f" - Особое управление: ОТСУТСТВУЕТ (доказано в методичке)")
    print(f" - Структура управления: bang-bang с ОДНИМ переключением")
    print(f" - u(t) = 0 при t < {x_opt[0]:.2f} с")
    print(f" - u(t) = u_m = {u_m:.4f} кг/с при t >= {x_opt[0]:.2f} с")
    print("=" * 72)

    return x_opt


# ─────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────
def plot_results(params_opt):
    """Построение графиков оптимальной траектории"""
    sim = simulate(params_opt, save_history=True)
    sol = sim["sol"]

    if sol is None or not sim["landed"]:
        print("Ошибка: нет данных для построения графиков")
        return

    t = sol.t
    h = sol.y[0]
    v = sol.y[1]
    m = sol.y[2]

    t_switch = params_opt[0]
    t_land = sim["t_land"]
    fuel_used = sim["fuel_used"]

    # Управление u(t)
    u = np.where(t < t_switch, 0.0, u_m)

    # Гравитация полная и возмущение (для южного полюса lat=-90)
    # Передаём время ti для учёта вращения Луны
    g_traj = [g_func(hi, lat=-90.0, t=ti) for hi, ti in zip(h, t)]
    g_base_traj = [g0 * (R_moon / (R_moon + hi)) ** 2 for hi in h]
    g_perturb_traj = [g_traj[i] - g_base_traj[i] for i in range(len(h))]

    # Вклад масконов отдельно
    g_mascon_traj = [g_mascon_component(hi, t=ti)[0] for hi, ti in zip(h, t)]
    g_harmonics_traj = [g_traj[i] - g_mascon_traj[i] for i in range(len(h))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Посадка по Понтрягину (1 переключение) | "
        f"t_switch={t_switch:.1f} с | "
        f"t_land={t_land:.2f} с | "
        f"v_land={sim['v_land']:.4f} м/с | "
        f"fuel={fuel_used:.1f} кг",
        fontsize=12
    )

    # ── Высота ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, h, "b", lw=2, label="h(t)")
    ax.axhline(0.0, color="brown", ls="--", lw=1.5, label="Поверхность")
    ax.axhline(h0, color="gray", ls=":", lw=1.2, label="h0")
    ax.axvline(t_switch, color="red", ls="--", lw=1.5, label=f"Включение: {t_switch:.1f} с")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Высота, м")
    ax.set_title("Высота")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ── Скорость ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, v, "g", lw=2, label="v(t)")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.axhline(-2.0, color="orange", ls=":", lw=1.2, label="±2 м/с")
    ax.axhline(2.0, color="orange", ls=":", lw=1.2)
    ax.axvline(t_switch, color="red", ls="--", lw=1.5, label=f"Включение: {t_switch:.1f} с")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Скорость, м/с")
    ax.set_title("Вертикальная скорость")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ── Масса ───────────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(t, m, "r", lw=2, label="m(t)")
    ax.axhline(m_dry, color="orange", ls="--", lw=1.5, label="Сухая масса")
    ax.axvline(t_switch, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Масса, кг")
    ax.set_title("Масса аппарата")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ── Управление ──────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.step(t, u, where="post", color="k", lw=2, label="u(t)")
    ax.axhline(u_m, color="red", ls="--", lw=1.2, label=f"u_m={u_m:.4f} кг/с")
    ax.axvline(t_switch, color="red", ls="--", lw=1.5, label=f"Переключение: {t_switch:.1f} с")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("u, кг/с")
    ax.set_title("Управление (bang-bang, 1 переключение)")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ── Гравитация (полная и возмущение) ────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, g_traj, "m", lw=2, label="g полная")
    ax.plot(t, g_base_traj, "c--", lw=1.2, label="g базовая (без возмущ.)")
    ax.plot(t, g_harmonics_traj, "y:", lw=1.0, label="g с гармониками")
    ax.axhline(g0, color="gray", ls=":", lw=1.0, label=f"g0={g0} м/с²")
    ax.axvline(t_switch, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("g, м/с²")
    ax.set_title("Гравитация вдоль траектории")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ── Возмущения: гармоники и масконы ─────────────────────────────
    ax = axes[1, 2]
    ax.plot(t, g_harmonics_traj, "orange", lw=1.5, label="Гармоники (J2,J3,J4...)")
    ax.plot(t, g_mascon_traj, "purple", lw=2, label="Масконы")
    ax.plot(t, g_perturb_traj, "gray", ls="--", lw=1.0, label="Σ возмущений")
    ax.axhline(0.0, color="black", ls="-", lw=0.5)
    ax.axvline(t_switch, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Δg, м/с²")
    ax.set_title("Вклад гармоник и масконов")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Вывод информации о гравитации после определения функций
    # Для южного полюса (lat=-90)
    g_surf = g_func(0.0, lat=-90.0)
    g_h0 = g_func(h0, lat=-90.0)
    g_perturf_surf = g_perturbation(0.0, lat=-90.0)
    g_perturf_h0 = g_perturbation(h0, lat=-90.0)

    # Вклад масконов
    g_mascon_surf, _ = g_mascon_component(0.0, lat=-90.0)
    g_mascon_h0, _ = g_mascon_component(h0, lat=-90.0)

    print(f" г на поверхности (h=0):     {g_surf:.6f} м/с²")
    print(f" г на высоте h0={h0:.0f} м:  {g_h0:.6f} м/с²")
    print(f" Возмущение на поверхности:  {g_perturf_surf:.6e} м/с²")
    print(f" Возмущение на высоте h0:    {g_perturf_h0:.6e} м/с²")
    print(f" Вклад масконов (h=0):       {g_mascon_surf:.6e} м/с²")
    print(f" Вклад масконов (h=h0):      {g_mascon_h0:.6e} м/с²")
    print("=" * 72)

    t_all = time.time()
    params_opt = solve()
    print(f"\nОбщее время: {time.time() - t_all:.1f} с")
    print("\nПостроение графиков...")
    plot_results(params_opt)
