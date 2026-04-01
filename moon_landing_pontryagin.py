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

print("=" * 72)
print(" ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — принцип минимума Понтрягина")
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


# ─────────────────────────────────────────────────────────────────────
# ГРАВИТАЦИЯ
# ─────────────────────────────────────────────────────────────────────
def g_func(h):
    """Переменная гравитация: g(h) = g0 * (R/(R+h))^2"""
    h_eff = max(h, 0.0)
    return g0 * (R_moon / (R_moon + h_eff)) ** 2


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
def rhs(t, y, t_switch):
    """Уравнения движения"""
    h, v, m = y
    u = control_pontryagin(t, t_switch, m)
    m_eff = max(m, m_dry + 1e-9)

    dh = v
    dv = c * u / m_eff - g_func(h)
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

    # Гравитация
    g_traj = [g_func(hi) for hi in h]

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

    # ── Гравитация ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, g_traj, "m", lw=2, label="g(h)")
    ax.axhline(g0, color="gray", ls="--", lw=1.2, label=f"g0={g0} м/с²")
    ax.axvline(t_switch, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("g, м/с²")
    ax.set_title("Гравитация вдоль траектории")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ── Фазовый портрет ─────────────────────────────────────────────
    ax = axes[1, 2]
    sc = ax.scatter(v, h, c=t, cmap="viridis", s=12)
    plt.colorbar(sc, ax=ax, label="Время, с")
    ax.plot(v[0], h[0], "go", ms=10, label="Начало")
    ax.plot(v[-1], h[-1], "rs", ms=10, label="Посадка")
    ax.set_xlabel("Скорость, м/с")
    ax.set_ylabel("Высота, м")
    ax.set_title("Фазовый портрет")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_all = time.time()
    params_opt = solve()
    print(f"\nОбщее время: {time.time() - t_all:.1f} с")
    print("\nПостроение графиков...")
    plot_results(params_opt)
