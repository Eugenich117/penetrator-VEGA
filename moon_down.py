# -*- coding: utf-8 -*-
"""
========================================================================
ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — двухимпульсное управление с дросселем
========================================================================

Управление:
   u(t) = alpha1 * u_m,  если t_on1 <= t <= T1   (первый импульс)
   u(t) = alpha2 * u_m,  если t_on2 <= t <= T2   (второй импульс)
   u(t) = 0               во всех остальных случаях

   alpha1, alpha2 in [0.05, 1.0] — коэффициенты дросселя

Оптимизируются 6 параметров:
   t_on1  — начало первого импульса
   T1     — конец первого импульса
   alpha1 — тяга первого импульса  [0.05 … 1.0]
   t_on2  — начало второго импульса
   T2     — конец второго импульса (примерно момент касания)
   alpha2 — тяга второго импульса  [0.05 … 1.0]

Порядковое ограничение: t_on1 <= T1 <= t_on2 <= T2
========================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import multiprocessing as mp

# ─────────────────────────────────────────────────────────────────────
# ИСХОДНЫЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────
R_moon    = 1_737_000.0
g0        = 1.62
h0        = 15_000.0
v0        = -20.0
m0        = 5_560.0
m_dry     = 3_740.0
fuel0     = m0 - m_dry
c         = 3_050.0
P_max     = 20_000.0
u_m       = P_max / c          # кг/с
H_SLOW    = 50.0               # м — высота "медленной" зоны

# Временные ограничения
T_MIN  = 50.0
T_MAX  = 380.0
SIM_MAX = 600.0

# Ограничения против взлёта
ASCENT_TOL   = 2.0    # м
UPWARD_V_TOL = 0.2    # м/с

# Веса штрафов
W_TOUCH_V      = 1.0e6
W_NOT_LANDED   = 1.0e9
W_ASCENT       = 1.0e5
W_UPWARD_V     = 5.0e4
W_ALIGN        = 1.0e3
W_FUEL         = 1.0
W_UNDERGROUND  = 1.0e8
W_LOW_V        = 2.0e5

print("=" * 72)
print(" ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — двухимпульсный дроссель")
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
    h2, v2, m2 = h + 0.5*dt*k1[0], v + 0.5*dt*k1[1], m + 0.5*dt*k1[2]
    k2 = rhs(h2, v2, m2, u)
    h3, v3, m3 = h + 0.5*dt*k2[0], v + 0.5*dt*k2[1], m + 0.5*dt*k2[2]
    k3 = rhs(h3, v3, m3, u)
    h4, v4, m4 = h + dt*k3[0], v + dt*k3[1], m + dt*k3[2]
    k4 = rhs(h4, v4, m4, u)
    h_new = h + dt/6.0*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    v_new = v + dt/6.0*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    m_new = m + dt/6.0*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return h_new, v_new, max(m_new, m_dry)


# ─────────────────────────────────────────────────────────────────────
# УПРАВЛЕНИЕ С ДРОССЕЛЕМ
# params = [t_on1, T1, alpha1, t_on2, T2, alpha2]
# ─────────────────────────────────────────────────────────────────────
def control_bb(params, t, m):
    t_on1, T1, alpha1, t_on2, T2, alpha2 = params
    if m <= m_dry:
        return 0.0
    if t_on1 <= t <= T1:
        return float(alpha1) * u_m
    if t_on2 <= t <= T2:
        return float(alpha2) * u_m
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# МОДЕЛИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────
def simulate_bb(params, save_history=False):
    t_on1, T1, alpha1, t_on2, T2, alpha2 = map(float, params)

    h, v, m, t = h0, v0, m0, 0.0
    landed     = False
    underground = False
    max_h      = h0
    max_upward_v = max(0.0, v0)
    low_alt_v_penalty = 0.0

    t_land = h_land = v_land = m_land = None

    if save_history:
        hist_t = [t]; hist_h = [h]; hist_v = [v]
        hist_m = [m]; hist_u = [control_bb(params, t, m)]
        hist_g = [g_func(h)]

    while t < SIM_MAX:
        if h < 10.0:
            dt = 0.005
        elif h < 100.0:
            dt = 0.02
        elif h < 500.0:
            dt = 0.05
        else:
            dt = 0.2
        dt = min(dt, SIM_MAX - t)

        u = control_bb(params, t, m)
        h_prev, v_prev, m_prev, t_prev = h, v, m, t
        h, v, m = rk4_step(h, v, m, u, dt)
        t += dt

        max_h = max(max_h, h)
        max_upward_v = max(max_upward_v, v)

        if h < H_SLOW and v < 0.0:
            low_alt_v_penalty += v**2 * (H_SLOW - h) / H_SLOW * dt

        # Детектор касания
        if h_prev > 0.0 and h <= 0.0:
            alpha   = np.clip(h_prev / (h_prev - h), 0.0, 1.0)
            t_land  = t_prev + alpha * dt
            h_land  = 0.0
            v_land  = v_prev + alpha * (v - v_prev)
            m_land  = m_prev + alpha * (m - m_prev)
            landed  = True
            if save_history:
                hist_t.append(t_land); hist_h.append(h_land)
                hist_v.append(v_land); hist_m.append(max(m_land, m_dry))
                hist_u.append(control_bb(params, t_land, max(m_land, m_dry)))
                hist_g.append(g_func(h_land))
            break

        if h < -50.0:
            underground = True
            break

        if save_history:
            hist_t.append(t); hist_h.append(h); hist_v.append(v)
            hist_m.append(m); hist_u.append(control_bb(params, t, m))
            hist_g.append(g_func(h))

    if not landed:
        t_land, h_land, v_land, m_land = t, h, v, m

    m_land     = max(m_land, m_dry)
    fuel_used  = m0 - m_land
    ascent_amount = max(0.0, max_h - h0)

    result = {
        "landed":             landed,
        "underground":        underground,
        "t_land":             t_land,
        "h_land":             h_land,
        "v_land":             v_land,
        "m_land":             m_land,
        "fuel_used":          fuel_used,
        "max_h":              max_h,
        "ascent_amount":      ascent_amount,
        "max_upward_v":       max_upward_v,
        "t_minus_T2":         t_land - T2 if landed else SIM_MAX - T2,
        "low_alt_v_penalty":  low_alt_v_penalty,
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
    t_on1, T1, alpha1, t_on2, T2, alpha2 = map(float, params)

    if not np.all(np.isfinite(params)):
        return 1e15

    # Порядковое ограничение
    if not (0.0 <= t_on1 <= T1 <= t_on2 <= T2):
        return 1e15
    if T2 < T_MIN or T2 > T_MAX:
        return 1e15
    # Минимальный зазор между импульсами
    if t_on2 - T1 < 2.0:
        return 1e15

    sim = simulate_bb(params, save_history=False)

    ascent_excess  = max(0.0, sim["ascent_amount"] - ASCENT_TOL)
    upward_v_excess = max(0.0, sim["max_upward_v"] - UPWARD_V_TOL)

    J = 0.0
    J += W_FUEL    * sim["fuel_used"]
    J += W_ASCENT  * ascent_excess**2
    J += W_UPWARD_V * upward_v_excess**2
    J += W_ALIGN   * sim["t_minus_T2"]**2
    J += W_LOW_V   * sim["low_alt_v_penalty"]

    if sim["underground"]:
        J += W_UNDERGROUND

    if sim["landed"]:
        J += W_TOUCH_V * sim["v_land"]**2
    else:
        h_pen = max(sim["h_land"], 0.0) / h0
        v_pen = sim["v_land"] / 100.0
        J += W_NOT_LANDED
        J += 1.0e4 * h_pen**2
        J += 1.0e4 * v_pen**2

    return float(J)


# ─────────────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────
def solve_bb():
    # Границы: [t_on1, T1, alpha1, t_on2, T2, alpha2]
    bounds = [
        (  0.0, 200.0),   # t_on1
        (  5.0, 250.0),   # T1
        (  0.05,  1.0),   # alpha1  — дроссель 1-го импульса
        ( 10.0, 300.0),   # t_on2
        ( 50.0, 380.0),   # T2
        (  0.05,  1.0),   # alpha2  — дроссель 2-го импульса
    ]

    print("\nШаг 1: Глобальный поиск (Differential Evolution, 6 параметров) ...")
    t0 = time.time()
    ctx  = mp.get_context("fork")
    pool = ctx.Pool(processes=mp.cpu_count())
    res_de = differential_evolution(
        objective_bb,
        bounds=bounds,
        strategy="best1bin",
        maxiter=300,
        popsize=30,
        tol=1e-7,
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
    print(f" DE завершён за {time.time() - t0:.1f} с")
    print(f" J_de  = {res_de.fun:.6e}")
    x = res_de.x
    print(f" x_de  = [t_on1={x[0]:.2f}, T1={x[1]:.2f}, a1={x[2]:.3f},"
          f" t_on2={x[3]:.2f}, T2={x[4]:.2f}, a2={x[5]:.3f}]")

    print("\nШаг 2: Локальная полировка (L-BFGS-B) ...")
    t1 = time.time()
    res_local = minimize(
        objective_bb,
        x0=res_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2000,
            "ftol":    1e-15,
            "gtol":    1e-10,
            "maxls":   60,
            "eps":     0.01,
        }
    )
    print(f" Local завершён за {time.time() - t1:.1f} с")
    print(f" J_opt = {res_local.fun:.6e}")

    x_opt = res_local.x
    t_on1, T1, alpha1, t_on2, T2, alpha2 = x_opt
    sim   = simulate_bb(x_opt, save_history=False)

    print("\n" + "=" * 72)
    print(" РЕЗУЛЬТАТ ОПТИМИЗАЦИИ С ДРОССЕЛЕМ")
    print("=" * 72)
    print(f" Успешное касание   = {sim['landed']}")
    print(f" t_on1              = {t_on1:.4f} с")
    print(f" T1                 = {T1:.4f} с    | длительность 1 = {T1-t_on1:.3f} с")
    print(f" alpha1             = {alpha1:.4f}   | тяга 1 = {alpha1*P_max:.1f} Н  ({alpha1*100:.1f}%)")
    print(f" t_on2              = {t_on2:.4f} с    | пауза          = {t_on2-T1:.3f} с")
    print(f" T2                 = {T2:.4f} с    | длительность 2 = {T2-t_on2:.3f} с")
    print(f" alpha2             = {alpha2:.4f}   | тяга 2 = {alpha2*P_max:.1f} Н  ({alpha2*100:.1f}%)")
    print(f" t_land             = {sim['t_land']:.4f} с")
    print(f" h_land             = {sim['h_land']:.6f} м")
    print(f" v_land             = {sim['v_land']:.6f} м/с")
    print(f" m_land             = {sim['m_land']:.4f} кг")
    print(f" fuel_used          = {sim['fuel_used']:.4f} кг")
    print(f" max_h              = {sim['max_h']:.4f} м")
    print("=" * 72)

    return x_opt


# ─────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────
def plot_results(params_opt):
    sim = simulate_bb(params_opt, save_history=True)

    t  = sim["hist_t"]
    h  = sim["hist_h"]
    v  = sim["hist_v"]
    m  = sim["hist_m"]
    u  = sim["hist_u"]

    t_on1, T1, alpha1, t_on2, T2, alpha2 = params_opt
    t_land    = sim["t_land"]
    fuel_used = sim["fuel_used"]

    P1 = alpha1 * P_max
    P2 = alpha2 * P_max

    def vlines(ax):
        ax.axvline(t_on1, color="red",     ls="--", lw=1.2, label=f"Вкл 1 ({P1:.0f} Н)")
        ax.axvline(T1,    color="darkred", ls=":",  lw=1.2, label="Выкл 1")
        ax.axvline(t_on2, color="blue",    ls="--", lw=1.2, label=f"Вкл 2 ({P2:.0f} Н)")
        ax.axvline(T2,    color="navy",    ls=":",  lw=1.2, label="Выкл 2")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Посадка с дросселем | "
        f"[{t_on1:.1f}…{T1:.1f}с, {alpha1*100:.0f}%] + [{t_on2:.1f}…{T2:.1f}с, {alpha2*100:.0f}%] | "
        f"t_land={t_land:.2f}с | v_land={sim['v_land']:.4f} м/с | fuel={fuel_used:.1f} кг",
        fontsize=11
    )

    # ── Высота ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, h, "b", lw=2)
    ax.axhline(0.0, color="brown", ls="--", lw=1.5, label="Поверхность")
    ax.axhline(h0,  color="gray",  ls=":",  lw=1.2, label="h0")
    vlines(ax)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Высота, м")
    ax.set_title("Высота"); ax.grid(True); ax.legend(fontsize=7)

    # ── Скорость ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, v, "g", lw=2)
    ax.axhline(0.0,  color="gray",   ls="--", lw=1.0)
    ax.axhline(-2.0, color="orange", ls=":",  lw=1.2, label="±2 м/с")
    ax.axhline( 2.0, color="orange", ls=":",  lw=1.2)
    vlines(ax)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Скорость, м/с")
    ax.set_title("Вертикальная скорость"); ax.grid(True); ax.legend(fontsize=7)

    # ── Масса ───────────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(t, m, "r", lw=2)
    ax.axhline(m_dry, color="orange", ls="--", lw=1.5, label="Сухая масса")
    vlines(ax)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Масса, кг")
    ax.set_title("Масса аппарата"); ax.grid(True); ax.legend(fontsize=7)

    # ── Управление (тяга) ────────────────────────────────────────────
    ax = axes[1, 0]
    # Переводим расход в тягу для наглядности
    thrust = u * c
    ax.step(t, thrust, where="post", color="k", lw=2, label="Тяга, Н")
    ax.axhline(P_max,           color="red",    ls="--", lw=1.2, label=f"P_max={P_max:.0f} Н")
    ax.axhline(alpha1 * P_max,  color="red",    ls=":",  lw=1.0, label=f"P1={P1:.0f} Н ({alpha1*100:.0f}%)")
    ax.axhline(alpha2 * P_max,  color="blue",   ls=":",  lw=1.0, label=f"P2={P2:.0f} Н ({alpha2*100:.0f}%)")
    vlines(ax)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Тяга, Н")
    ax.set_title("Управление — тяга двигателя"); ax.grid(True); ax.legend(fontsize=7)

    # ── Скорость zoom (h < 500 м) ────────────────────────────────────
    ax = axes[1, 1]
    mask = h < 500.0
    ax.plot(t[mask], v[mask], "g", lw=2)
    ax.axhline( 0.0, color="gray",   ls="--", lw=1.0)
    ax.axhline(-2.0, color="orange", ls=":",  lw=1.2, label="±2 м/с (допуск)")
    ax.axhline( 2.0, color="orange", ls=":",  lw=1.2)
    vlines(ax)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Скорость, м/с")
    ax.set_title("Скорость (h < 500 м)"); ax.grid(True); ax.legend(fontsize=7)

    # ── Фазовый портрет ─────────────────────────────────────────────
    ax = axes[1, 2]
    sc = ax.scatter(v, h, c=t, cmap="viridis", s=12)
    plt.colorbar(sc, ax=ax, label="Время, с")
    ax.plot(v[0], h[0], "go", ms=8, label="Начало")
    ax.plot(v[-1], h[-1], "rs", ms=8, label="Касание")
    ax.axvline(-2.0, color="orange", ls=":", lw=1.2, label="±2 м/с")
    ax.axvline( 2.0, color="orange", ls=":", lw=1.2)
    ax.set_xlabel("Скорость, м/с"); ax.set_ylabel("Высота, м")
    ax.set_title("Фазовый портрет"); ax.grid(True); ax.legend(fontsize=7)

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