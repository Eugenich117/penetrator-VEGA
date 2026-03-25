# -*- coding: utf-8 -*-
"""
=======================================================================
ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — УСТОЙЧИВАЯ ВЕРСИЯ БЕЗ "ВЗЛЁТА НА 6 КМ"
=======================================================================

Что сделано:
1. Полностью убран косвенный метод стрельбы.
2. Используется прямой метод:
   - управление параметризуется 10 узлами тяги alpha_i in [0, 1]
   - между узлами линейная интерполяция
   - оптимизация сначала DE, потом локальная полировка L-BFGS-B
3. Жёстко штрафуются:
   - подъём выше начальной высоты h0
   - положительная вертикальная скорость (взлёт)
   - недолёт до поверхности
   - большая скорость в момент касания
4. Интегрирование с event-детектором касания h=0.
5. Нормальные графики без mathtext-ошибок.

Идея:
эта версия не "доказывает" Понтрягина, зато обычно решает задачу
устойчиво и без идиотских траекторий "подпрыгнуть на 6 км".
=======================================================================
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
u_m    = P_max / c                    # кг/с

# Горизонт моделирования
T_MAX = 300.0                         # верхняя граница времени моделирования
N_STEPS = 500                         # шагов интегрирования
DT = T_MAX / N_STEPS

# Параметризация управления
N_NODES = 10                          # число узлов управления alpha(t) ∈ [0,1]

# Ограничения против "взлёта"
ASCENT_TOL = 5.0                      # допустимый микроподъём, м
UPWARD_V_TOL = 0.5                    # допустимая положительная скорость, м/с

# Веса штрафов
W_H_NOT_LANDED = 5e6
W_V_NOT_LANDED = 2e5
W_V_TOUCH      = 3e4
W_ASCENT       = 3e6
W_UPWARD_V     = 5e5
W_SMOOTH       = 50.0
W_TIME         = 1.0

print("=" * 72)
print(" ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — устойчивая прямая оптимизация")
print("=" * 72)
print(f" h0      = {h0:.1f} м")
print(f" v0      = {v0:.1f} м/с")
print(f" m0      = {m0:.1f} кг")
print(f" m_dry   = {m_dry:.1f} кг")
print(f" fuel0   = {fuel0:.1f} кг")
print(f" c       = {c:.1f} м/с")
print(f" P_max   = {P_max:.1f} Н")
print(f" u_m     = {u_m:.5f} кг/с")
print(f" T_MAX   = {T_MAX:.1f} с")
print(f" N_NODES = {N_NODES}")
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
    u_eff = 0.0 if m <= m_dry else max(0.0, min(u, u_m))

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
# УПРАВЛЕНИЕ
# ─────────────────────────────────────────────────────────────────────
TAU_NODES = np.linspace(0.0, 1.0, N_NODES)

def control_profile(alpha_nodes, tau):
    alpha = np.interp(tau, TAU_NODES, alpha_nodes)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * u_m


# ─────────────────────────────────────────────────────────────────────
# МОДЕЛИРОВАНИЕ ТРАЕКТОРИИ
# ─────────────────────────────────────────────────────────────────────
def simulate(alpha_nodes, save_history=False):
    h = h0
    v = v0
    m = m0
    t = 0.0

    max_h = h
    max_upward_v = max(v, 0.0)

    landed = False
    t_land = None
    h_land = None
    v_land = None
    m_land = None

    if save_history:
        hist_t = [t]
        hist_h = [h]
        hist_v = [v]
        hist_m = [m]
        hist_u = [control_profile(alpha_nodes, 0.0)]
        hist_g = [g_func(h)]

    for k in range(N_STEPS):
        tau = min(t / T_MAX, 1.0)
        u = control_profile(alpha_nodes, tau)

        h_prev, v_prev, m_prev, t_prev = h, v, m, t
        h, v, m = rk4_step(h, v, m, u, DT)
        t += DT

        max_h = max(max_h, h)
        max_upward_v = max(max_upward_v, v)

        # Детектор касания поверхности
        if h_prev > 0.0 and h <= 0.0:
            alpha = h_prev / (h_prev - h)
            alpha = max(0.0, min(1.0, alpha))

            t_land = t_prev + alpha * DT
            h_land = 0.0
            v_land = v_prev + alpha * (v - v_prev)
            m_land = m_prev + alpha * (m - m_prev)
            landed = True

            if save_history:
                hist_t.append(t_land)
                hist_h.append(h_land)
                hist_v.append(v_land)
                hist_m.append(max(m_land, m_dry))
                hist_u.append(u)
                hist_g.append(g_func(h_land))
            break

        if save_history:
            hist_t.append(t)
            hist_h.append(h)
            hist_v.append(v)
            hist_m.append(m)
            hist_u.append(u)
            hist_g.append(g_func(h))

    if not landed:
        t_land = t
        h_land = h
        v_land = v
        m_land = m

    result = {
        "landed": landed,
        "t_land": t_land,
        "h_land": h_land,
        "v_land": v_land,
        "m_land": max(m_land, m_dry),
        "fuel_used": m0 - max(m_land, m_dry),
        "max_h": max_h,
        "ascent_amount": max(0.0, max_h - h0),
        "max_upward_v": max_upward_v,
        "h_end": h,
        "v_end": v,
        "m_end": m,
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
# ФУНКЦИЯ ЦЕЛИ
# ─────────────────────────────────────────────────────────────────────
def objective(alpha_nodes):
    alpha_nodes = np.clip(np.asarray(alpha_nodes, dtype=float), 0.0, 1.0)
    sim = simulate(alpha_nodes, save_history=False)

    fuel_used = sim["fuel_used"]
    ascent_excess = max(0.0, sim["ascent_amount"] - ASCENT_TOL)
    upward_v_excess = max(0.0, sim["max_upward_v"] - UPWARD_V_TOL)
    smooth_pen = np.mean(np.diff(alpha_nodes)**2)

    J = 0.0
    J += fuel_used
    J += W_ASCENT * ascent_excess**2
    J += W_UPWARD_V * upward_v_excess**2
    J += W_SMOOTH * smooth_pen
    J += W_TIME * sim["t_land"]

    if sim["landed"]:
        J += W_V_TOUCH * sim["v_land"]**2
    else:
        J += 1e6
        J += W_H_NOT_LANDED * (max(sim["h_land"], 0.0) / h0)**2
        J += W_V_NOT_LANDED * (sim["v_land"] / 100.0)**2

    return float(J)


# ─────────────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────
def solve():
    print("\nШаг 1: Глобальный поиск (Differential Evolution)...")
    bounds = [(0.0, 1.0)] * N_NODES

    t0 = time.time()
    res_de = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=120,
        popsize=18,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.85,
        seed=42,
        polish=False,
        disp=False,
        workers=1,
    )
    print(f" DE завершён за {time.time() - t0:.1f} с")
    print(f" J_de = {res_de.fun:.6e}")

    print("\nШаг 2: Локальная полировка (L-BFGS-B)...")
    t1 = time.time()
    res_local = minimize(
        objective,
        x0=res_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
            "maxls": 50
        }
    )
    print(f" Local завершён за {time.time() - t1:.1f} с")
    print(f" J_opt = {res_local.fun:.6e}")

    alpha_opt = np.clip(res_local.x, 0.0, 1.0)
    sim_opt = simulate(alpha_opt, save_history=False)

    print("\n" + "=" * 72)
    print(" РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
    print("=" * 72)
    print(f" Приземление            = {sim_opt['landed']}")
    print(f" Время касания          = {sim_opt['t_land']:.3f} с")
    print(f" Высота в касании       = {sim_opt['h_land']:.6f} м")
    print(f" Скорость в касании     = {sim_opt['v_land']:.6f} м/с")
    print(f" Конечная масса         = {sim_opt['m_land']:.3f} кг")
    print(f" Расход топлива         = {sim_opt['fuel_used']:.3f} кг")
    print(f" Макс. высота           = {sim_opt['max_h']:.3f} м")
    print(f" Подъём над h0          = {sim_opt['ascent_amount']:.6f} м")
    print(f" Макс. положит. скорость= {sim_opt['max_upward_v']:.6f} м/с")
    print("=" * 72)
    print(" alpha_opt =", np.array2string(alpha_opt, precision=4, separator=", "))
    print("=" * 72)

    return alpha_opt


# ─────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────
def plot_results(alpha_opt):
    sim = simulate(alpha_opt, save_history=True)

    t = sim["hist_t"]
    h = sim["hist_h"]
    v = sim["hist_v"]
    m = sim["hist_m"]
    u = sim["hist_u"]
    g = sim["hist_g"]

    T_land = sim["t_land"]
    v_land = sim["v_land"]
    fuel_used = sim["fuel_used"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Оптимальная посадка на Луну | t_land = {T_land:.2f} c | "
        f"v_land = {v_land:.3f} м/с | fuel = {fuel_used:.1f} кг",
        fontsize=13
    )

    ax = axes[0, 0]
    ax.plot(t, h, "b", lw=2)
    ax.axhline(0.0, color="brown", ls="--", lw=1.5, label="Поверхность")
    ax.axhline(h0, color="gray", ls=":", lw=1.2, label="Начальная высота")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Высота, м")
    ax.set_title("Высота")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, v, "g", lw=2)
    ax.axhline(0.0, color="gray", ls="--", lw=1.2)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Скорость, м/с")
    ax.set_title("Вертикальная скорость")
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(t, m, "r", lw=2)
    ax.axhline(m_dry, color="orange", ls="--", lw=1.5, label="Сухая масса")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Масса, кг")
    ax.set_title("Масса аппарата")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    ax.step(t, u, where="post", color="k", lw=2)
    ax.axhline(u_m, color="red", ls="--", lw=1.2, label=f"u_m={u_m:.3f}")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Расход топлива u, кг/с")
    ax.set_title("Управление")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, g, color="purple", lw=2)
    ax.axhline(g0, color="gray", ls="--", lw=1.0, label=f"g0={g0}")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("g(h), м/с²")
    ax.set_title("Гравитация")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 2]
    sc = ax.scatter(v, h, c=t, cmap="viridis", s=12)
    plt.colorbar(sc, ax=ax, label="Время, с")
    ax.plot(v[0], h[0], "go", ms=8, label="Начало")
    ax.plot(v[-1], h[-1], "rs", ms=8, label="Касание")
    ax.set_xlabel("Скорость, м/с")
    ax.set_ylabel("Высота, м")
    ax.set_title("Фазовый портрет")
    ax.grid(True)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_all = time.time()

    alpha_opt = solve()

    print(f"\nОбщее время: {time.time() - t_all:.1f} с")
    print("\nПостроение графиков...")
    plot_results(alpha_opt)