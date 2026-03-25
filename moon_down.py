# -*- coding: utf-8 -*-
"""
========================================================================
ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — bang-bang версия
========================================================================

Что изменено:
1. Управление только типа bang-bang:
      u(t) = 0,   t < t_on
      u(t) = u_m, t_on <= t <= T
      u(t) = 0,   t > T
   На оптимуме обычно получается T ≈ t_land, то есть по сути одно включение.

2. Оптимизируются только 2 параметра:
      t_on  — момент включения двигателя
      T     — конец активного участка / целевой момент посадки

3. Целевая функция:
   - сильно штрафует ненулевую скорость в касании
   - штрафует недолёт / перелёт
   - запрещает "взлёт вверх"
   - слабо минимизирует расход топлива

4. Интегрирование с детектором касания h = 0.

Это уже не косвенный метод Понтрягина, а устойчивый прямой поиск
по малому числу параметров управления.
========================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# ─────────────────────────────────────────────────────────────────────
# ИСХОДНЫЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────
R_moon = 1_737_000.0
g0     = 1.62
h0     = 15_000.0
v0     = -20.0
m0     = 5_560.0
m_dry  = 3_740.0
fuel0  = m0 - m_dry
c      = 3_050.0
P_max  = 20_000.0
u_m    = P_max / c  # кг/с

# Временные ограничения
T_MIN = 50.0
T_MAX = 350.0
SIM_MAX = 500.0

# Ограничения против "взлёта"
ASCENT_TOL = 2.0      # м
UPWARD_V_TOL = 0.2    # м/с

# Веса штрафов
W_TOUCH_V = 2.0e6
W_NOT_LANDED_H = 5.0e7
W_NOT_LANDED_V = 1.0e6
W_ASCENT = 2.0e7
W_UPWARD_V = 2.0e6
W_ALIGN = 5.0e4
W_FUEL = 1.0
W_UNDERGROUND = 1.0e8

print("=" * 72)
print(" ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — bang-bang версия")
print("=" * 72)
print(f" h0      = {h0:.1f} м")
print(f" v0      = {v0:.1f} м/с")
print(f" m0      = {m0:.1f} кг")
print(f" m_dry   = {m_dry:.1f} кг")
print(f" fuel0   = {fuel0:.1f} кг")
print(f" c       = {c:.1f} м/с")
print(f" P_max   = {P_max:.1f} Н")
print(f" u_m     = {u_m:.6f} кг/с")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────
# ГРАВИТАЦИЯ
# ─────────────────────────────────────────────────────────────────────
def g_func(h):
    h_eff = max(h, 0.0)
    return g0 * (R_moon / (R_moon + h_eff)) ** 2


# ─────────────────────────────────────────────────────────────────────
# ДИНАМИКА
# ─────────────────────────────────────────────────────────────────────
def rhs(h, v, m, u):
    m_eff = max(m, m_dry)
    u_eff = 0.0 if m <= m_dry else np.clip(u, 0.0, u_m)

    dh = v
    dv = c * u_eff / m_eff - g_func(h)
    dm = -u_eff
    return dh, dv, dm


def rk4_step(h, v, m, u, dt):
    k1 = rhs(h, v, m, u)

    h2 = h + 0.5 * dt * k1[0]
    v2 = v + 0.5 * dt * k1[1]
    m2 = m + 0.5 * dt * k1[2]
    k2 = rhs(h2, v2, m2, u)

    h3 = h + 0.5 * dt * k2[0]
    v3 = v + 0.5 * dt * k2[1]
    m3 = m + 0.5 * dt * k2[2]
    k3 = rhs(h3, v3, m3, u)

    h4 = h + dt * k3[0]
    v4 = v + dt * k3[1]
    m4 = m + dt * k3[2]
    k4 = rhs(h4, v4, m4, u)

    h_new = h + dt / 6.0 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    v_new = v + dt / 6.0 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    m_new = m + dt / 6.0 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

    if m_new < m_dry:
        m_new = m_dry

    return h_new, v_new, m_new


# ─────────────────────────────────────────────────────────────────────
# BANG-BANG УПРАВЛЕНИЕ
# ─────────────────────────────────────────────────────────────────────
def control_bb(params, t, m):
    t_on, T = params
    if m <= m_dry:
        return 0.0
    if t < t_on:
        return 0.0
    if t <= T:
        return u_m
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# МОДЕЛИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────
def simulate_bb(params, save_history=False):
    t_on, T = float(params[0]), float(params[1])

    h = h0
    v = v0
    m = m0
    t = 0.0

    landed = False
    underground = False

    max_h = h0
    max_upward_v = max(0.0, v0)

    t_land = None
    h_land = None
    v_land = None
    m_land = None

    if save_history:
        hist_t = [t]
        hist_h = [h]
        hist_v = [v]
        hist_m = [m]
        hist_u = [control_bb(params, t, m)]
        hist_g = [g_func(h)]

    while t < SIM_MAX:
        dt = 0.05 if h < 300.0 else 0.2
        dt = min(dt, SIM_MAX - t)

        u = control_bb(params, t, m)

        h_prev, v_prev, m_prev, t_prev = h, v, m, t
        h, v, m = rk4_step(h, v, m, u, dt)
        t += dt

        max_h = max(max_h, h)
        max_upward_v = max(max_upward_v, v)

        # Детектор касания поверхности
        if h_prev > 0.0 and h <= 0.0:
            alpha = h_prev / (h_prev - h)
            alpha = np.clip(alpha, 0.0, 1.0)

            t_land = t_prev + alpha * dt
            h_land = 0.0
            v_land = v_prev + alpha * (v - v_prev)
            m_land = m_prev + alpha * (m - m_prev)
            landed = True

            if save_history:
                hist_t.append(t_land)
                hist_h.append(h_land)
                hist_v.append(v_land)
                hist_m.append(max(m_land, m_dry))
                hist_u.append(control_bb(params, t_land, max(m_land, m_dry)))
                hist_g.append(g_func(h_land))
            break

        if h < -50.0:
            underground = True
            break

        if save_history:
            hist_t.append(t)
            hist_h.append(h)
            hist_v.append(v)
            hist_m.append(m)
            hist_u.append(control_bb(params, t, m))
            hist_g.append(g_func(h))

    if not landed:
        t_land = t
        h_land = h
        v_land = v
        m_land = m

    m_land = max(m_land, m_dry)
    fuel_used = m0 - m_land
    ascent_amount = max(0.0, max_h - h0)

    result = {
        "landed": landed,
        "underground": underground,
        "t_land": t_land,
        "h_land": h_land,
        "v_land": v_land,
        "m_land": m_land,
        "fuel_used": fuel_used,
        "max_h": max_h,
        "ascent_amount": ascent_amount,
        "max_upward_v": max_upward_v,
        "t_minus_T": t_land - T if landed else SIM_MAX - T,
    }

    if save_history:
        result["hist_t"] = np.array(hist_t)
        result["hist_h"] = np.array(hist_h)
        result["hist_v"] = np.array(hist_v)
        result["hist_m"] = np.array(hist_m)
        result["hist_u"] = np.array(hist_u)
        result["hist_g"] = np.array(hist_g)

    return result


# ─────────────────────────────────────────────────────────────────────
# ЦЕЛЕВАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────
def objective_bb(params):
    t_on, T = float(params[0]), float(params[1])

    # Базовые ограничения
    if not np.isfinite(t_on) or not np.isfinite(T):
        return 1e15
    if t_on < 0.0 or T < T_MIN or T > T_MAX:
        return 1e15
    if t_on > T:
        return 1e15

    sim = simulate_bb(params, save_history=False)

    ascent_excess = max(0.0, sim["ascent_amount"] - ASCENT_TOL)
    upward_v_excess = max(0.0, sim["max_upward_v"] - UPWARD_V_TOL)

    J = 0.0

    # Экономим топливо с маленьким весом — главное сначала посадить
    J += W_FUEL * sim["fuel_used"]

    # Запрещаем "подпрыгивания"
    J += W_ASCENT * ascent_excess**2
    J += W_UPWARD_V * upward_v_excess**2

    # Желательно, чтобы T совпадал с реальным касанием
    J += W_ALIGN * sim["t_minus_T"]**2

    if sim["underground"]:
        J += W_UNDERGROUND

    if sim["landed"]:
        J += W_TOUCH_V * sim["v_land"]**2
        # лёгкий штраф, если касание получилось с слишком большим запасом до T или после T
        J += 1e4 * (sim["h_land"]**2)
    else:
        h_pen = max(sim["h_land"], 0.0) / h0
        v_pen = sim["v_land"] / 100.0
        J += 1e9
        J += W_NOT_LANDED_H * h_pen**2
        J += W_NOT_LANDED_V * v_pen**2

    return float(J)


# ─────────────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────
def solve_bb():
    print("\nШаг 1: Глобальный поиск по [t_on, T] (Differential Evolution)...")
    bounds = [
        (0.0, 260.0),   # t_on
        (60.0, 320.0),  # T
    ]

    t0 = time.time()
    res_de = differential_evolution(
        objective_bb,
        bounds=bounds,
        strategy="best1bin",
        maxiter=120,
        popsize=18,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.85,
        seed=42,
        polish=False,
        disp=False,
        workers=1,
    )
    print(f" DE завершён за {time.time() - t0:.1f} с")
    print(f" J_de = {res_de.fun:.6e}")
    print(f" x_de = [t_on={res_de.x[0]:.6f}, T={res_de.x[1]:.6f}]")

    print("\nШаг 2: Локальная полировка (L-BFGS-B)...")
    t1 = time.time()
    res_local = minimize(
        objective_bb,
        x0=res_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 1000,
            "ftol": 1e-14,
            "maxls": 50,
        }
    )
    print(f" Local завершён за {time.time() - t1:.1f} с")
    print(f" J_opt = {res_local.fun:.6e}")

    x_opt = res_local.x
    sim = simulate_bb(x_opt, save_history=False)

    print("\n" + "=" * 72)
    print(" РЕЗУЛЬТАТ bang-bang ОПТИМИЗАЦИИ")
    print("=" * 72)
    print(f" Успешное касание         = {sim['landed']}")
    print(f" t_on                     = {x_opt[0]:.6f} с")
    print(f" T                        = {x_opt[1]:.6f} с")
    print(f" t_land                   = {sim['t_land']:.6f} с")
    print(f" T - t_land               = {x_opt[1] - sim['t_land']:.6f} с")
    print(f" h_land                   = {sim['h_land']:.9f} м")
    print(f" v_land                   = {sim['v_land']:.9f} м/с")
    print(f" m_land                   = {sim['m_land']:.6f} кг")
    print(f" fuel_used                = {sim['fuel_used']:.6f} кг")
    print(f" max_h                    = {sim['max_h']:.6f} м")
    print(f" ascent_amount            = {sim['ascent_amount']:.6f} м")
    print(f" max_upward_v             = {sim['max_upward_v']:.6f} м/с")
    print("=" * 72)

    return x_opt


# ─────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────
def plot_results(params_opt):
    sim = simulate_bb(params_opt, save_history=True)

    t = sim["hist_t"]
    h = sim["hist_h"]
    v = sim["hist_v"]
    m = sim["hist_m"]
    u = sim["hist_u"]
    g = sim["hist_g"]

    t_on, T = params_opt
    t_land = sim["t_land"]
    fuel_used = sim["fuel_used"]

    print("\n" + "=" * 72)
    print(" ДИАГНОСТИКА ТРАЕКТОРИИ")
    print("=" * 72)
    print(f" t_on          = {t_on:.6f} с")
    print(f" T             = {T:.6f} с")
    print(f" t_land        = {t_land:.6f} с")
    print(f" h_land        = {sim['h_land']:.9f} м")
    print(f" v_land        = {sim['v_land']:.9f} м/с")
    print(f" fuel_used     = {fuel_used:.6f} кг")
    print(f" max_h         = {sim['max_h']:.6f} м")
    print(f" ascent_amount = {sim['ascent_amount']:.6f} м")
    print("=" * 72)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Bang-bang посадка на Луну | t_on={t_on:.2f} c | "
        f"T={T:.2f} c | t_land={t_land:.2f} c | "
        f"v_land={sim['v_land']:.4f} м/с | fuel={fuel_used:.1f} кг",
        fontsize=13
    )

    # Высота
    ax = axes[0, 0]
    ax.plot(t, h, "b", lw=2)
    ax.axhline(0.0, color="brown", ls="--", lw=1.5, label="Поверхность")
    ax.axhline(h0, color="gray", ls=":", lw=1.2, label="Начальная высота")
    ax.axvline(t_on, color="red", ls="--", lw=1.2, label="Включение")
    ax.axvline(T, color="black", ls=":", lw=1.2, label="T")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Высота, м")
    ax.set_title("Высота")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Скорость
    ax = axes[0, 1]
    ax.plot(t, v, "g", lw=2)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.axvline(t_on, color="red", ls="--", lw=1.2)
    ax.axvline(T, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Скорость, м/с")
    ax.set_title("Вертикальная скорость")
    ax.grid(True)

    # Масса
    ax = axes[0, 2]
    ax.plot(t, m, "r", lw=2)
    ax.axhline(m_dry, color="orange", ls="--", lw=1.5, label="Сухая масса")
    ax.axvline(t_on, color="red", ls="--", lw=1.2)
    ax.axvline(T, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Масса, кг")
    ax.set_title("Масса аппарата")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Управление
    ax = axes[1, 0]
    ax.step(t, u, where="post", color="k", lw=2)
    ax.axhline(u_m, color="red", ls="--", lw=1.2, label=f"u_m={u_m:.3f}")
    ax.axvline(t_on, color="red", ls="--", lw=1.2)
    ax.axvline(T, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("u, кг/с")
    ax.set_title("Bang-bang управление")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Гравитация
    ax = axes[1, 1]
    ax.plot(t, g, color="purple", lw=2)
    ax.axhline(g0, color="gray", ls="--", lw=1.0, label=f"g0={g0}")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("g(h), м/с²")
    ax.set_title("Гравитация")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Фазовый портрет
    ax = axes[1, 2]
    sc = ax.scatter(v, h, c=t, cmap="viridis", s=12)
    plt.colorbar(sc, ax=ax, label="Время, с")
    ax.plot(v[0], h[0], "go", ms=8, label="Начало")
    ax.plot(v[-1], h[-1], "rs", ms=8, label="Касание")
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

    params_opt = solve_bb()

    print(f"\nОбщее время: {time.time() - t_all:.1f} с")
    print("\nПостроение графиков...")
    plot_results(params_opt)