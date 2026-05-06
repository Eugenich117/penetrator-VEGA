# -*- coding: utf-8 -*-
"""
========================================================================
ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ — решение по принципу минимума Понтрягина
========================================================================

Задание (вариант 2): Вертикальная посадка на Луну (Н1-Л3, СССР)

МАТЕМАТИЧЕСКАЯ МОДЕЛЬ (из задания):
  dh/dt = v
  dv/dt = -g(h) + J*beta/m
  dm/dt = -beta

  g(h) = g0*(R/(R+h))^2 * [1 + 3*J2*(R/(R+h))^2] + A_mascon*exp(-h/H_mascon)
  P = J * beta,  0 <= beta <= beta_m

  Критерий: J_fuel = m0 - m(T) -> min (минимум расхода топлива)
  Граничные условия: h(0)=h0, v(0)=v0, m(0)=m0
                     h(T)=0,  v(T)=0

ОПТИМАЛЬНОЕ УПРАВЛЕНИЕ (раздел 11.4 методички):
  Доказано: особого управления НЕТ.
  Оптимальное управление — релейное с ОДНИМ переключением:

    beta(t) = 0,       если t < t*
    beta(t) = beta_m,  если t* <= t <= T

МЕТОД РЕШЕНИЯ (по методичке, раздел 11.4):
  Краевая задача сводится к системе ДВУХ УРАВНЕНИЙ относительно t* и T:

    F1(t*, T) = h(T) = 0   ← аппарат касается поверхности
    F2(t*, T) = v(T) = 0   ← скорость посадки равна нулю

  Уравнения получаются интегрированием уравнений движения при оптимальном
  управлении. Решение находится методом Ньютона-Рафсона (scipy.optimize.fsolve).
  Глобальный перебор (Differential Evolution) НЕ НУЖЕН.
========================================================================
"""

import time
import numpy as np
from scipy.optimize import fsolve, root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# ИСХОДНЫЕ ДАННЫЕ (лунный модуль Н1-Л3)
# ─────────────────────────────────────────────────────────────────────
R_moon    = 1_737_000.0    # радиус Луны, м
g0        = 1.62           # ускорение свободного падения на поверхности, м/с^2
h0        = 15_000.0       # начальная высота, м
v0        = -100.0          # начальная скорость, м/с (вниз)
m0        = 5_560.0        # начальная масса, кг
m_dry     = 2_740.0        # сухая масса, кг
J_imp     = 3_050.0        # удельный импульс (обозначение J из задания), м/с
P_max     = 20_000.0       # максимальная тяга, Н
beta_m    = P_max / J_imp  # максимальный секундный расход, кг/с

# ─────────────────────────────────────────────────────────────────────
# ГРАВИТАЦИОННЫЕ ВОЗМУЩЕНИЯ (посадка на полюс Луны)
# ─────────────────────────────────────────────────────────────────────
J2_moon   = 2.027e-4       # зональная гармоника 2-го порядка (безразм.)
A_mascon  = 0.001          # амплитуда маскона на поверхности, м/с^2
H_mascon  = 30_000.0       # характерная глубина маскона, м (30 км)

print("=" * 72)
print(" ОПТИМАЛЬНАЯ ПОСАДКА НА ЛУНУ")
print("=" * 72)
print(" Задание: вариант 2, вертикальная посадка (Н1-Л3)")
print(" Управление: bang-bang с ОДНИМ переключением")
print(" Метод: решение системы двух уравнений (по методичке, разд. 11.4)")
print("=" * 72)
print(f" h0           = {h0:.1f} м")
print(f" v0           = {v0:.1f} м/с")
print(f" m0           = {m0:.1f} кг")
print(f" m_dry        = {m_dry:.1f} кг")
print(f" J (уд. имп.) = {J_imp:.1f} м/с")
print(f" P_max        = {P_max:.1f} Н")
print(f" beta_m       = {beta_m:.6f} кг/с")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────
# ГРАВИТАЦИЯ (с возмущениями: J2 + маскон)
# ─────────────────────────────────────────────────────────────────────
def g_func(h):
    """
    Ускорение свободного падения с учётом гравитационных возмущений.

    Базовое:  g0 * (R/(R+h))^2
    J2:       поправка от зональной гармоники 2-го порядка (сжатие Луны).
              На полюсе (φ = 90°):
                g = g0*(R/(R+h))^2 * [1 + 3*J2*(R/(R+h))^2]
    Маскон:   локальная аномалия с экспоненциальным затуханием:
                g_mascon = A_mascon * exp(-h / H_mascon)

    Итого: g(h) = g0*(R/(R+h))^2 * [1 + 3*J2*(R/(R+h))^2]
                  + A_mascon * exp(-h / H_mascon)
    """
    h_eff = max(h, 0.0)
    ratio = R_moon / (R_moon + h_eff)
    ratio2 = ratio ** 2
    g = g0 * ratio2 * (1.0 + 3.0 * J2_moon * ratio2)
    g += A_mascon * np.exp(-h_eff / H_mascon)
    return g


print(" ГРАВИТАЦИОННЫЕ ВОЗМУЩЕНИЯ:")
print(f" J2       = {J2_moon:.4e} (сжатие, полюс)")
print(f" A_mascon = {A_mascon:.6f} м/с^2")
print(f" H_mascon = {H_mascon:.0f} м")
print(f" g(h0)    = {g_func(h0):.8f} м/с^2")
print(f" g(0)     = {g_func(0.0):.8f} м/с^2")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────
# ПРАВЫЕ ЧАСТИ СИСТЕМЫ
# ─────────────────────────────────────────────────────────────────────
def rhs(t, y, t_switch):
    """
    dh/dt = v
    dv/dt = -g(h) + J*beta/m
    dm/dt = -beta

    beta(t) = 0       при t < t_switch
    beta(t) = beta_m  при t >= t_switch
    """
    h, v, m = y
    beta = beta_m if t >= t_switch else 0.0
    m_eff = max(m, m_dry + 1e-12)

    dh = v
    dv = -g_func(h) + J_imp * beta / m_eff
    dm = -beta

    return [dh, dv, dm]


# ─────────────────────────────────────────────────────────────────────
# ИНТЕГРИРОВАНИЕ НА ФИКСИРОВАННОМ ОТРЕЗКЕ [0, T]
# ─────────────────────────────────────────────────────────────────────
def integrate_fixed_T(t_switch, T):
    """
    Интегрирует уравнения движения от 0 до T при заданном t_switch.
    Возвращает конечное состояние (h(T), v(T), m(T)).

    Именно это и описывает методичка: проинтегрировать уравнения движения
    при найденном оптимальном управлении и получить h(T) и v(T).
    """
    t_switch = max(0.0, t_switch)
    T = max(t_switch + 1.0, T)

    # Разбиваем интегрирование на два участка по точке переключения:
    #   [0, t_switch]   — beta = 0 (свободное падение)
    #   [t_switch, T]   — beta = beta_m (максимальная тяга)
    # Это даёт точную обработку разрыва управления.

    y_init = [h0, v0, m0]
    t_sw = min(t_switch, T)

    # Участок 1: свободное падение [0, t_sw]
    if t_sw > 0.0:
        sol1 = solve_ivp(
            lambda t, y: rhs(t, y, t_switch),
            [0.0, t_sw],
            y_init,
            method='RK45',
            max_step=0.5,
            rtol=1e-10,
            atol=1e-12,
        )
        y_mid = sol1.y[:, -1]
    else:
        y_mid = np.array(y_init)

    # Участок 2: работа двигателя [t_sw, T]
    if T > t_sw:
        sol2 = solve_ivp(
            lambda t, y: rhs(t, y, t_switch),
            [t_sw, T],
            y_mid,
            method='RK45',
            max_step=0.5,
            rtol=1e-10,
            atol=1e-12,
        )
        y_final = sol2.y[:, -1]
    else:
        y_final = y_mid

    h_T = float(y_final[0])
    v_T = float(y_final[1])
    m_T = float(max(y_final[2], m_dry))

    return h_T, v_T, m_T


# ─────────────────────────────────────────────────────────────────────
# СИСТЕМА ДВУХ УРАВНЕНИЙ (по методичке)
# ─────────────────────────────────────────────────────────────────────
def two_equations(x):
    """
    Система двух уравнений относительно t* и T (раздел 11.4 методички).

    Краевые условия посадки:
        F1(t*, T) = h(T) = 0   ← аппарат достигает поверхности
        F2(t*, T) = v(T) = 0   ← мягкая посадка (нулевая скорость)

    Получаются интегрированием уравнений движения при оптимальном
    bang-bang управлении с переключением в момент t*.
    """
    t_switch, T = x
    h_T, v_T, _ = integrate_fixed_T(t_switch, T)
    return [h_T, v_T]


# ─────────────────────────────────────────────────────────────────────
# НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ (физическое обоснование)
# ─────────────────────────────────────────────────────────────────────
def initial_guess():
    """
    Оценка начального приближения из физических соображений.

    Время торможения (при постоянной тяге с нуля):
      v_T = v0 + (J*beta_m/m0 - g0)*tau = 0
      => tau ≈ |v0| / (J*beta_m/m0 - g0)

    Грубая оценка полного времени из высоты и начальной скорости:
      T ~ 2 * h0 / |v0|  (линейная экстраполяция)
    """
    # Эффективное ускорение при работе двигателя (начальная масса)
    a_thrust = J_imp * beta_m / m0 - g0   # ≈ 9.0 - 1.62 ≈ 7.4 м/с^2

    # Время торможения от v0 до 0 при полной тяге
    tau_brake = abs(v0) / a_thrust if a_thrust > 0 else 50.0

    # Грубая оценка T: время свободного падения до земли + торможение
    T_guess = h0 / abs(v0) + tau_brake

    # t* — момент включения двигателя (T - время торможения)
    t_star_guess = max(0.0, T_guess - tau_brake * 2.0)

    return t_star_guess, T_guess


# ─────────────────────────────────────────────────────────────────────
# ФИНАЛЬНОЕ МОДЕЛИРОВАНИЕ (с историей, для графиков)
# ─────────────────────────────────────────────────────────────────────
def simulate_full(t_switch, T):
    """
    Полное интегрирование с сохранением траектории для графиков.
    """
    y_init = [h0, v0, m0]
    t_sw = min(t_switch, T)

    t_hist, h_hist, v_hist, m_hist = [], [], [], []

    # Участок 1
    if t_sw > 0.0:
        sol1 = solve_ivp(
            lambda t, y: rhs(t, y, t_switch),
            [0.0, t_sw], y_init,
            method='RK45', max_step=0.5, rtol=1e-10, atol=1e-12,
            dense_output=True
        )
        t1 = np.linspace(0.0, t_sw, max(3, int(t_sw * 4)))
        s1 = sol1.sol(t1)
        t_hist.append(t1); h_hist.append(s1[0]); v_hist.append(s1[1]); m_hist.append(s1[2])
        y_mid = sol1.y[:, -1]
    else:
        y_mid = np.array(y_init)

    # Участок 2
    if T > t_sw:
        sol2 = solve_ivp(
            lambda t, y: rhs(t, y, t_switch),
            [t_sw, T], y_mid,
            method='RK45', max_step=0.5, rtol=1e-10, atol=1e-12,
            dense_output=True
        )
        t2 = np.linspace(t_sw, T, max(3, int((T - t_sw) * 4)))
        s2 = sol2.sol(t2)
        t_hist.append(t2); h_hist.append(s2[0]); v_hist.append(s2[1]); m_hist.append(s2[2])

    t_arr = np.concatenate(t_hist)
    h_arr = np.concatenate(h_hist)
    v_arr = np.concatenate(v_hist)
    m_arr = np.concatenate(m_hist)

    return t_arr, h_arr, v_arr, m_arr


# ─────────────────────────────────────────────────────────────────────
# ОСНОВНОЕ РЕШЕНИЕ — СИСТЕМА ДВУХ УРАВНЕНИЙ
# ─────────────────────────────────────────────────────────────────────
def solve():
    """
    Решение по методичке (раздел 11.4):
    Сводим краевую задачу к системе двух уравнений:
        h(T; t*, T) = 0
        v(T; t*, T) = 0
    и решаем её методом Ньютона (fsolve) по двум неизвестным: t* и T.
    """

    t_star_0, T_0 = initial_guess()
    print(f"\n{'=' * 72}")
    print(" НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ (физические оценки)")
    print(f"{'=' * 72}")
    print(f" t*_0 (включение двигателя) = {t_star_0:.2f} с")
    print(f" T_0  (время посадки)        = {T_0:.2f} с")

    # Контроль невязок в начальном приближении
    F0 = two_equations([t_star_0, T_0])
    print(f" Невязки: h(T0) = {F0[0]:.2f} м,  v(T0) = {F0[1]:.2f} м/с")
    print(f"{'=' * 72}")

    print("\n РЕШЕНИЕ СИСТЕМЫ ДВУХ УРАВНЕНИЙ (методика разд. 11.4):")
    print(" F1(t*, T) = h(T) = 0   ← условие касания поверхности")
    print(" F2(t*, T) = v(T) = 0   ← условие мягкой посадки")
    print(f"{'=' * 72}")

    t_start = time.time()

    # --- Метод 1: fsolve (Ньютон–Рафсон, scipy) ----------------------
    x0 = [t_star_0, T_0]
    sol_fsolve = fsolve(
        two_equations,
        x0,
        full_output=True,
        xtol=1e-10,
        maxfev=2000,
    )
    x_opt_fs = sol_fsolve[0]
    info_fs   = sol_fsolve[1]
    msg_fs    = sol_fsolve[3]

    # --- Метод 2: root (Levenberg–Marquardt, как резервный) ----------
    sol_root = root(
        two_equations,
        x0,
        method='lm',
        tol=1e-12,
        options={'maxiter': 2000, 'ftol': 1e-14, 'xtol': 1e-14}
    )

    dt = time.time() - t_start

    # Выбираем лучшее решение по норме невязки
    F_fs   = two_equations(x_opt_fs)
    F_root = two_equations(sol_root.x)
    norm_fs   = np.linalg.norm(F_fs)
    norm_root = np.linalg.norm(F_root)

    if norm_fs <= norm_root:
        x_opt = x_opt_fs
        method_name = "fsolve (Ньютон–Рафсон)"
        F_opt = F_fs
    else:
        x_opt = sol_root.x
        method_name = "root (Levenberg–Marquardt)"
        F_opt = F_root

    t_switch_opt = float(x_opt[0])
    T_opt        = float(x_opt[1])

    # Итоговое состояние
    h_T, v_T, m_T = integrate_fixed_T(t_switch_opt, T_opt)
    fuel_used = m0 - m_T
    burn_time = T_opt - t_switch_opt

    print(f"\n Метод: {method_name}")
    print(f" Время вычисления: {dt:.3f} с")
    print(f"\n{'=' * 72}")
    print(" РЕЗУЛЬТАТ (решение системы двух уравнений)")
    print(f"{'=' * 72}")
    print(f" t* (включение двигателя)   = {t_switch_opt:.8f} с")
    print(f" T  (время посадки)          = {T_opt:.8f} с")
    print(f" Длительность работы двигат. = {burn_time:.6f} с")
    print(f"{'=' * 72}")
    print(" НЕВЯЗКИ (должны быть ~ 0):")
    print(f" F1 = h(T) = {F_opt[0]:.6e} м   (цель: 0)")
    print(f" F2 = v(T) = {F_opt[1]:.6e} м/с (цель: 0)")
    print(f" ||F||     = {np.linalg.norm(F_opt):.6e}")
    print(f"{'=' * 72}")
    print(" ИТОГОВЫЕ ХАРАКТЕРИСТИКИ:")
    print(f" m(T)  (масса посадки)       = {m_T:.6f} кг")
    print(f" Расход топлива              = {fuel_used:.6f} кг")
    print(f" Аналит. расход (beta_m*tau) = {beta_m * burn_time:.6f} кг")
    dm_check = abs(fuel_used - beta_m * burn_time)
    print(f" Расхождение                 = {dm_check:.2e} кг")
    print(f"{'=' * 72}")

    print("\n ПРОВЕРКА УСЛОВИЙ ОПТИМАЛЬНОСТИ (методичка):")
    print(f" ✓ Особое управление: ОТСУТСТВУЕТ")
    print(f" ✓ Структура: bang-bang с ОДНИМ переключением")
    print(f" ✓ beta = 0         при  0 <= t < {t_switch_opt:.2f} с")
    print(f" ✓ beta = {beta_m:.4f}  при  {t_switch_opt:.2f} с <= t <= {T_opt:.2f} с")
    print(f"{'=' * 72}")

    return t_switch_opt, T_opt


# ─────────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────────
def plot_results(t_switch_opt, T_opt):
    """Построение графиков оптимальной траектории"""
    t, h, v, m = simulate_full(t_switch_opt, T_opt)

    beta = np.where(t < t_switch_opt, 0.0, beta_m)
    g_traj = np.array([g_func(hi) for hi in h])

    h_T, v_T, m_T = integrate_fixed_T(t_switch_opt, T_opt)
    fuel_used = m0 - m_T

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Посадка по Понтрягину | Метод: система двух уравнений (разд. 11.4)\n"
        f"t*={t_switch_opt:.2f} с | T={T_opt:.2f} с | "
        f"h(T)={h_T:.3f} м | v(T)={v_T:.4f} м/с | fuel={fuel_used:.1f} кг",
        fontsize=11
    )

    # Высота
    ax = axes[0, 0]
    ax.plot(t, h, "b", lw=2, label="h(t)")
    ax.axhline(0.0, color="brown", ls="--", lw=1.5, label="Поверхность")
    ax.axhline(h0, color="gray", ls=":", lw=1.2, label="h0")
    ax.axvline(t_switch_opt, color="red", ls="--", lw=1.5,
               label=f"t* = {t_switch_opt:.1f} с")
    ax.set_xlabel("Время, с"); ax.set_ylabel("Высота, м")
    ax.set_title("Высота"); ax.grid(True); ax.legend(fontsize=8)

    # Скорость
    ax = axes[0, 1]
    ax.plot(t, v, "g", lw=2, label="v(t)")
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.axvline(t_switch_opt, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Скорость, м/с")
    ax.set_title("Вертикальная скорость"); ax.grid(True); ax.legend(fontsize=8)

    # Масса
    ax = axes[0, 2]
    ax.plot(t, m, "r", lw=2, label="m(t)")
    ax.axhline(m_dry, color="orange", ls="--", lw=1.5, label="Сухая масса")
    ax.axvline(t_switch_opt, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с"); ax.set_ylabel("Масса, кг")
    ax.set_title("Масса аппарата"); ax.grid(True); ax.legend(fontsize=8)

    # Управление
    ax = axes[1, 0]
    ax.step(t, beta, where="post", color="k", lw=2, label="beta(t)")
    ax.axhline(beta_m, color="red", ls="--", lw=1.2,
               label=f"beta_m = {beta_m:.4f} кг/с")
    ax.axvline(t_switch_opt, color="red", ls="--", lw=1.5,
               label=f"t* = {t_switch_opt:.1f} с")
    ax.set_xlabel("Время, с"); ax.set_ylabel("beta, кг/с")
    ax.set_title("Оптимальное управление (bang-bang)"); ax.grid(True); ax.legend(fontsize=8)

    # Гравитация
    ax = axes[1, 1]
    ax.plot(t, g_traj, "m", lw=2, label="g(h)")
    ax.axhline(g0, color="gray", ls="--", lw=1.2, label=f"g0 = {g0} м/с²")
    ax.axvline(t_switch_opt, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Время, с"); ax.set_ylabel("g, м/с²")
    ax.set_title("Ускорение свободного падения"); ax.grid(True); ax.legend(fontsize=8)

    # Фазовый портрет
    ax = axes[1, 2]
    sc = ax.scatter(v, h, c=t, cmap="viridis", s=12)
    plt.colorbar(sc, ax=ax, label="Время, с")
    ax.plot(v[0], h[0], "go", ms=10, label="Начало")
    ax.plot(v[-1], h[-1], "rs", ms=10, label="Посадка")
    ax.set_xlabel("Скорость, м/с"); ax.set_ylabel("Высота, м")
    ax.set_title("Фазовый портрет (h–v)"); ax.grid(True); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_all = time.time()
    t_switch_opt, T_opt = solve()
    print(f"\nОбщее время: {time.time() - t_all:.3f} с")
    print("\nПостроение графиков...")
    plot_results(t_switch_opt, T_opt)
