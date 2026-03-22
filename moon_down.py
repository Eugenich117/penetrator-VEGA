"""
=======================================================================
 КУРСОВАЯ РАБОТА — ВАРИАНТ 2
 Оптимальная вертикальная посадка лунного модуля (Н1-Л3)

 Постановка:
   ẋ₁ = x₂                       x₁ — высота, м
   ẋ₂ = c·u/x₃ − g               x₂ — вертикальная скорость, м/с
   ẋ₃ = −u                        x₃ — масса КА, кг
   ψ̇₁ = 0
   ψ̇₂ = −ψ₁
   ψ̇₃ = ψ₂·c·u / x₃²

 Управление (bang-bang):
   Q = ψ₂·c/x₃ − ψ₃
   u = u_m  если Q < 0,  u = 0  если Q ≥ 0

 Критерий: J = ∫u dt → min  (минимум расхода топлива)
 Терминальные условия: x₁(T) = 0, x₂(T) = 0, ψ₃(T) = −1

 Остановка интегратора: событие x₁ = 0
 Искомые: ψ₁(0), ψ₂(0), ψ₃(0)  — три неизвестных начальных условия
=======================================================================
"""

import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
#  ИСХОДНЫЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────
# Луна
g   = 1.62          # ускорение свободного падения, м/с²

# Лунный модуль Н1-Л3 (ЛК) — открытые источники
h0      = 15_000.0   # начальная высота, м
dh0     =  -20.0     # начальная вертикальная скорость, м/с (вниз)
m0      =  5_560.0   # начальная масса, кг
m_dry   =  3_740.0   # сухая масса (без топлива), кг
mf      = m0 - m_dry # запас топлива, кг
c       =  3_050.0   # удельный импульс, м/с  (≈ 311 с · g₀)
P_max   = 20_000.0   # максимальная тяга, Н
u_m     = P_max / c  # максимальный расход, кг/с  ≈ 6.56 кг/с

# Начальные условия вектора состояния
X0 = [h0, dh0, m0]

# Весовые коэффициенты невязки (можно менять)
lam1 = 1.0        # вес для x₁(T)
lam2 = 1.0        # вес для x₂(T)
lam3 = 1.0        # вес для ψ₃(T)+1

DT   = 1.0        # базовый шаг РК4, с
DT_FINE = 0.05    # уменьшенный шаг вблизи поверхности (h < 300 м)

print('=' * 62)
print(' ВАРИАНТ 2: Оптимальная посадка на Луну, Н1-Л3 (ЛК)')
print('=' * 62)
print(f' h₀     = {h0} м')
print(f' ẋ₂(0)  = {dh0} м/с')
print(f' m₀     = {m0} кг')
print(f' m_dry  = {m_dry} кг  (запас топлива = {mf:.0f} кг)')
print(f' c      = {c} м/с')
print(f' P_max  = {P_max} Н  →  u_m = {u_m:.4f} кг/с')
print(f' g      = {g} м/с²')
print('=' * 62)


# ─────────────────────────────────────────────────────────────────
#  ФУНКЦИЯ РК4 — без изменений из исходного кода проекта
# ─────────────────────────────────────────────────────────────────
def runge_kutta_4(equations, initial, dt, dx):
    k1 = {key: 0 for key in dx}
    k2 = {key: 0 for key in dx}
    k3 = {key: 0 for key in dx}
    k4 = {key: 0 for key in dx}

    for eq in equations:
        derivative, key = eq(initial)
        if key in dx:
            k1[key] = derivative

    state_k2 = initial.copy()
    for key in dx:
        state_k2[key] += dt / 2 * k1[key]
    for eq in equations:
        derivative, key = eq(state_k2)
        if key in dx:
            k2[key] = derivative

    state_k3 = initial.copy()
    for key in dx:
        state_k3[key] += dt / 2 * k2[key]
    for eq in equations:
        derivative, key = eq(state_k3)
        if key in dx:
            k3[key] = derivative

    state_k4 = initial.copy()
    for key in dx:
        state_k4[key] += dt * k3[key]
    for eq in equations:
        derivative, key = eq(state_k4)
        if key in dx:
            k4[key] = derivative

    new_values = []
    for key in dx:
        new_val = initial[key] + (dt / 6) * (
            k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]
        )
        new_values.append(new_val)
    return new_values


# ─────────────────────────────────────────────────────────────────
#  УПРАВЛЕНИЕ
# ─────────────────────────────────────────────────────────────────
def Q_switch(state):
    """Функция переключения Q = ψ₂·c/x₃ − ψ₃"""
    return state['psi2'] * c / state['x3'] - state['psi3']

def control(state):
    return 0.0 if Q_switch(state) >= 0 else u_m


# ─────────────────────────────────────────────────────────────────
#  ПРАВЫЕ ЧАСТИ КАНОНИЧЕСКОЙ СИСТЕМЫ
# ─────────────────────────────────────────────────────────────────
def dx1_eq(state):
    return state['x2'], 'x1'

def dx2_eq(state):
    u = control(state)
    return c * u / state['x3'] - g, 'x2'

def dx3_eq(state):
    return -control(state), 'x3'

def dpsi1_eq(state):
    return 0.0, 'psi1'

def dpsi2_eq(state):
    return -state['psi1'], 'psi2'

def dpsi3_eq(state):
    u = control(state)
    return c * u * state['psi2'] / (state['x3'] ** 2), 'psi3'


EQUATIONS = [dx1_eq, dx2_eq, dx3_eq, dpsi1_eq, dpsi2_eq, dpsi3_eq]
DX_KEYS   = ['x1', 'x2', 'x3', 'psi1', 'psi2', 'psi3']

# максимальное время интегрирования (защита от зависания)
T_MAX = 5000.0


# ─────────────────────────────────────────────────────────────────
#  ИНТЕГРИРОВАНИЕ: остановка по событию x₁ = 0
# ─────────────────────────────────────────────────────────────────
def integrate(psi0, save_history=False):
    """
    Интегрирует каноническую систему до x₁ = 0 (посадка).
    Возвращает финальное состояние словарём.
    Если save_history=True — также историю по времени.
    """
    state = {
        'x1':   X0[0],
        'x2':   X0[1],
        'x3':   X0[2],
        'psi1': psi0[0],
        'psi2': psi0[1],
        'psi3': psi0[2],
    }

    if save_history:
        hist   = {k: [state[k]] for k in DX_KEYS}
        t_hist = [0.0]
        u_hist = [control(state)]
        Q_hist = [Q_switch(state)]

    t = 0.0
    while t < T_MAX:
        h_cur = state['x1']

        # Адаптивный шаг: уменьшаем вблизи поверхности
        if h_cur < 300.0:
            dt = DT_FINE
        else:
            dt = DT

        # Не перешагнём через h=0: ограничиваем шаг по Эйлеру
        # (приближение момента касания)
        v_cur = state['x2']
        if v_cur < 0 and h_cur > 0:
            t_touch = -h_cur / v_cur   # линейная оценка времени до h=0
            if t_touch < dt:
                dt = max(t_touch * 0.5, 1e-4)

        vals = runge_kutta_4(EQUATIONS, state, dt, DX_KEYS)

        # обновляем состояние
        state['x1']   = vals[0]
        state['x2']   = vals[1]
        state['x3']   = max(vals[2], m_dry)   # топливо не может стать < 0
        state['psi1'] = vals[3]
        state['psi2'] = vals[4]
        state['psi3'] = vals[5]
        t += dt

        if save_history:
            for k in DX_KEYS:
                hist[k].append(state[k])
            t_hist.append(t)
            u_hist.append(control(state))
            Q_hist.append(Q_switch(state))

        # Условие остановки: достигли поверхности
        if state['x1'] <= 0.0:
            state['x1'] = 0.0
            state['t']  = t
            break

        # Аварийная остановка: КА улетает вверх и не вернётся
        if state['x1'] > h0 * 3:
            state['t'] = t
            break
    else:
        state['t'] = T_MAX

    if save_history:
        return state, hist, t_hist, u_hist, Q_hist
    return state


# ─────────────────────────────────────────────────────────────────
#  ФУНКЦИЯ НЕВЯЗКИ
# ─────────────────────────────────────────────────────────────────
def residual(params):
    """
    z = λ₁·x₁²(T) + λ₂·x₂²(T) + λ₃·(ψ₃(T)+1)²
    Минимум z = 0 соответствует точному выполнению
    терминальных условий.
    """
    state = integrate(params)
    z = (lam1 * state['x1'] ** 2 +
         lam2 * state['x2'] ** 2 +
         lam3 * (state['psi3'] + 1.0) ** 2)
    return z


# ─────────────────────────────────────────────────────────────────
#  ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────
def solve():
    print('\nШаг 1: Глобальный поиск (дифференциальная эволюция)...')

    # Диапазоны начальных условий сопряжённого вектора
    bounds = [
        (-0.1,  0.1),    # ψ₁(0)
        (-1.0,  0.0),    # ψ₂(0)
        (-3.0,  0.0),    # ψ₃(0)
    ]

    t0 = time.time()
    res_global = differential_evolution(
        residual,
        bounds,
        maxiter=500,
        tol=1e-10,
        seed=42,
        popsize=15,
        mutation=(0.5, 1.5),
        recombination=0.9,
        workers=1,
        disp=False,
    )
    print(f'  Завершён за {time.time()-t0:.1f} с')
    print(f'  z_global = {res_global.fun:.6e}')
    print(f'  ψ(0) = {res_global.x}')

    print('\nШаг 2: Уточнение (Нелдер–Мид)...')
    t0 = time.time()
    res_local = minimize(
        residual,
        res_global.x,
        method='Nelder-Mead',
        options={
            'xatol':   1e-12,
            'fatol':   1e-14,
            'maxiter': 100_000,
            'adaptive': True,
        },
    )
    print(f'  Завершён за {time.time()-t0:.1f} с')
    print(f'  z_final  = {res_local.fun:.6e}')
    print(f'  ψ₁(0) = {res_local.x[0]:.8f}')
    print(f'  ψ₂(0) = {res_local.x[1]:.8f}')
    print(f'  ψ₃(0) = {res_local.x[2]:.8f}')
    return res_local


# ─────────────────────────────────────────────────────────────────
#  ГРАФИКИ
# ─────────────────────────────────────────────────────────────────
FONT = 'Times New Roman'
FS   = 14
plt.rcParams['font.family']      = FONT
plt.rcParams['axes.titlesize']   = 12
plt.rcParams['axes.labelsize']   = FS
plt.rcParams['mathtext.fontset'] = 'cm'   # Computer Modern — лучший LaTeX-рендер


def plot_results(psi0_opt):
    state, hist, t_hist, u_hist, Q_hist = integrate(
        psi0_opt, save_history=True)

    t  = np.array(t_hist)
    T  = state['t']
    fuel_used = m0 - state['x3']

    print('\n' + '=' * 62)
    print(' РЕЗУЛЬТАТЫ ОПТИМАЛЬНОГО РЕШЕНИЯ')
    print('=' * 62)
    print(f' Время посадки T        = {T:.2f} с')
    print(f' Высота в момент T      = {state["x1"]:.4f} м  (цель: 0)')
    print(f' Скорость в момент T    = {state["x2"]:.4f} м/с (цель: 0)')
    print(f' psi3(T)                = {state["psi3"]:.6f}  (цель: -1)')
    print(f' Конечная масса         = {state["x3"]:.2f} кг')
    print(f' Расход топлива         = {fuel_used:.2f} кг')
    print(f' Остаток топлива        = {state["x3"] - m_dry:.2f} кг')
    print('=' * 62)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.canvas.manager.set_window_title(
        'Оптимальная посадка — Н1-Л3 (вариант 2)')
    fig.suptitle(
        f'Оптимальная вертикальная посадка. Лунный модуль Н1-Л3\n'
        f'$T = {T:.1f}$ с,  расход топлива $= {fuel_used:.1f}$ кг',
        fontsize=14)

    # ── 1. Высота ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, hist['x1'], 'b', linewidth=2)
    ax.axhline(0, color='brown', linestyle='--', linewidth=1,
               label='Поверхность $(h = 0)$')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Высота $h$, м')
    ax.set_title('Высота от времени')
    ax.legend(fontsize=10)
    ax.grid(True)

    # ── 2. Скорость ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t, hist['x2'], 'g', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'Скорость $\dot{x}_2$, м/с')
    ax.set_title('Вертикальная скорость')
    ax.grid(True)

    # ── 3. Масса ──────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(t, hist['x3'], 'r', linewidth=2)
    ax.axhline(m_dry, color='orange', linestyle='--',
               linewidth=1.5, label=f'Сухая масса $= {m_dry}$ кг')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Масса $m$, кг')
    ax.set_title('Масса КА')
    ax.legend(fontsize=10)
    ax.grid(True)

    # ── 4. Управление u(t) ────────────────────────────────────────
    ax = axes[1, 0]
    ax.step(t, u_hist, 'k', linewidth=2, where='post')
    ax.axhline(u_m, color='red', linestyle='--',
               linewidth=1.5, label=f'$u_m = {u_m:.3f}$ кг/с')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('$u$, кг/с')
    ax.set_title(r'Управление $u(t)$ — расход топлива')
    ax.set_ylim(-0.1 * u_m, 1.2 * u_m)
    ax.legend(fontsize=10)
    ax.grid(True)

    # ── 5. Сопряжённые переменные ─────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, hist['psi1'], label=r'$\psi_1$', linewidth=2)
    ax.plot(t, hist['psi2'], label=r'$\psi_2$', linewidth=2)
    ax.plot(t, hist['psi3'], label=r'$\psi_3$', linewidth=2)
    ax.axhline(-1, color='gray', linestyle=':', linewidth=1.5,
               label=r'$\psi_3(T) = -1$')
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$\psi$')
    ax.set_title(r'Сопряжённые переменные $\psi(t)$')
    ax.legend(fontsize=10)
    ax.grid(True)

    # ── 6. Функция переключения Q(t) ──────────────────────────────
    ax = axes[1, 2]
    ax.plot(t, Q_hist, 'm', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5,
               label='$Q = 0$ — переключение')
    ax.fill_between(t, Q_hist, 0,
                    where=[q < 0 for q in Q_hist],
                    alpha=0.2, color='red', label='Двигатель ВКЛ')
    ax.fill_between(t, Q_hist, 0,
                    where=[q >= 0 for q in Q_hist],
                    alpha=0.2, color='blue', label='Двигатель ВЫКЛ')
    ax.set_xlabel('Время, с')
    ax.set_ylabel(r'$Q(x,\,\psi)$')
    ax.set_title(r'Функция переключения $Q(t)$')
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()

    # ── Фазовый портрет h vs V ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    fig2.canvas.manager.set_window_title('Фазовый портрет посадки')
    sc = ax2.scatter(hist['x2'], hist['x1'],
                     c=t, cmap='viridis', s=8)
    plt.colorbar(sc, ax=ax2, label='Время, с')
    ax2.plot(hist['x2'][0],  hist['x1'][0],  'go', markersize=12,
             label='Начало')
    ax2.plot(hist['x2'][-1], hist['x1'][-1], 'rs', markersize=12,
             label='Посадка')
    ax2.set_xlabel(r'Скорость $\dot{x}_2$, м/с')
    ax2.set_ylabel('Высота $h$, м')
    ax2.set_title(r'Фазовый портрет: $h$ vs $\dot{x}_2$')
    ax2.legend(fontsize=11)
    ax2.grid(True)
    plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────────────────────────
#  ГЛАВНАЯ ПРОГРАММА
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_total = time.time()

    res = solve()

    print(f'\nОбщее время: {time.time() - t_total:.1f} с')
    print('\nПостроение графиков...')
    plot_results(res.x)