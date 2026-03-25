"""
=======================================================================
ИСПРАВЛЕННАЯ ВЕРСИЯ — Оптимальная посадка на Луну

Ключевое исправление:
- integrate() останавливается при h=0 (касание поверхности) с
  линейной интерполяцией точного момента пересечения.
- T в параметрах — верхняя граница времени, а не точное время посадки.
- residual() штрафует:
    • скорость в момент касания  — v(t_land)²
    • высоту, если земля не достигнута за время T  — h(T)²
    • условие трансверсальности  — (ψ₃(T)+1)², H(T)²

Это исключает локальный минимум "v=0 при h=5 м".
=======================================================================
"""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as _mp

# ─────────────────────────────────────────────────────────────────
# ИСХОДНЫЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────
R_moon = 1_737_000.0
g0     = 1.62
h0     = 15_000.0
dh0    = -20.0
m0     = 5_560.0
m_dry  = 3_740.0
mf     = m0 - m_dry
c      = 3_050.0
P_max  = 20_000.0
u_m    = P_max / c

X0 = [h0, dh0, m0]

# Веса невязки
lam1 = 500.0    # штраф за высоту (если земля не достигнута)
lam2 = 300.0   # штраф за скорость при касании — доминирующий
lam3 = 1.0     # штраф за ψ₃(T)+1
lam4 = 1.0     # штраф за H(T) (трансверсальность по T)

DT      = 1.0
DT_FINE = 0.1
T_MAX   = 200.0   # фиксированная верхняя граница — НЕ оптимизируется

print('=' * 70)
print(' ИСПРАВЛЕННАЯ ВЕРСИЯ: Оптимальная посадка на Луну')
print('=' * 70)
print(f' h₀    = {h0} м,  ẋ₂(0) = {dh0} м/с,  m₀ = {m0} кг')
print(f' m_dry = {m_dry} кг  (топливо = {mf:.0f} кг)')
print(f' c     = {c} м/с,  P_max = {P_max} Н,  u_m = {u_m:.4f} кг/с')
print('=' * 70)

# ─────────────────────────────────────────────────────────────────
# ГРАВИТАЦИЯ
# ─────────────────────────────────────────────────────────────────
def g_func(h):
    return g0 * (R_moon / (R_moon + max(h, 0.0))) ** 2

def dg_dh(h):
    return -2.0 * g0 * R_moon**2 / (R_moon + max(h, 0.0))**3

# ─────────────────────────────────────────────────────────────────
# РК4
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
# УПРАВЛЕНИЕ
# ─────────────────────────────────────────────────────────────────
def Q_switch(state):
    return state['psi2'] * c / state['x3'] - state['psi3']

def control(state):
    if state['x3'] <= m_dry:
        return 0.0
    return 0.0 if Q_switch(state) >= 0 else u_m

# ─────────────────────────────────────────────────────────────────
# УРАВНЕНИЯ ДВИЖЕНИЯ И СОПРЯЖЁННОЙ СИСТЕМЫ
# ─────────────────────────────────────────────────────────────────
def dx1_eq(state):
    return state['x2'], 'x1'

def dx2_eq(state):
    u = control(state)
    return c * u / state['x3'] - g_func(state['x1']), 'x2'

def dx3_eq(state):
    return -control(state), 'x3'

def dpsi1_eq(state):
    return state['psi2'] * dg_dh(state['x1']), 'psi1'

def dpsi2_eq(state):
    return -state['psi1'], 'psi2'

def dpsi3_eq(state):
    u = control(state)
    return c * u * state['psi2'] / (state['x3'] ** 2), 'psi3'

def hamiltonian(state):
    u = control(state)
    return (state['psi1'] * state['x2']
            + state['psi2'] * (c * u / state['x3'] - g_func(state['x1']))
            - state['psi3'] * u)

EQUATIONS = [dx1_eq, dx2_eq, dx3_eq, dpsi1_eq, dpsi2_eq, dpsi3_eq]
DX_KEYS   = ['x1', 'x2', 'x3', 'psi1', 'psi2', 'psi3']

# ─────────────────────────────────────────────────────────────────
# ИНТЕГРИРОВАНИЕ С ДЕТЕКТОРОМ КАСАНИЯ ЗЕМЛИ
#
# Ключевое изменение: при пересечении h=0 делаем линейную
# интерполяцию, чтобы найти точный момент t_land и все
# переменные состояния в этот момент. Цикл завершается досрочно.
# Если земля не достигнута за T_final — возвращаем последнее
# состояние с флагом landed=False.
# ─────────────────────────────────────────────────────────────────
def integrate(psi0, T_final, save_history=False):
    state = {
        'x1':   X0[0],
        'x2':   X0[1],
        'x3':   X0[2],
        'psi1': psi0[0],
        'psi2': psi0[1],
        'psi3': psi0[2],
    }
    state['landed'] = False

    if save_history:
        hist   = {k: [state[k]] for k in DX_KEYS}
        t_hist = [0.0]
        u_hist = [control(state)]
        Q_hist = [Q_switch(state)]
        g_hist = [g_func(state['x1'])]

    t = 0.0
    while t < T_final:
        h_cur = state['x1']
        dt = DT_FINE if h_cur < 500.0 else DT
        dt = min(dt, T_final - t)

        # Сохраняем состояние ДО шага
        state_before = state.copy()
        t_before = t

        vals = runge_kutta_4(EQUATIONS, state, dt, DX_KEYS)

        # ── ДЕТЕКТОР КАСАНИЯ ──────────────────────────────────────
        # Если до шага h > 0, а после — h ≤ 0, интерполируем
        h_after = vals[0]
        if state_before['x1'] > 0.0 and h_after <= 0.0:
            # Линейная интерполяция: alpha — доля шага до h=0
            alpha = state_before['x1'] / (state_before['x1'] - h_after)
            alpha = max(0.0, min(1.0, alpha))

            keys_list = DX_KEYS
            prev_vals = [state_before[k] for k in keys_list]

            for i, key in enumerate(keys_list):
                state[key] = prev_vals[i] + alpha * (vals[i] - prev_vals[i])

            state['x1'] = 0.0                      # точно на поверхности
            state['x3'] = max(state['x3'], m_dry)
            t = t_before + alpha * dt
            state['landed'] = True

            if save_history:
                for k in DX_KEYS:
                    hist[k].append(state[k])
                t_hist.append(t)
                u_hist.append(control(state))
                Q_hist.append(Q_switch(state))
                g_hist.append(g_func(state['x1']))
            break
        # ─────────────────────────────────────────────────────────

        state['x1']   = vals[0]
        state['x2']   = vals[1]
        state['x3']   = max(vals[2], m_dry)
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
            g_hist.append(g_func(state['x1']))

    state['t'] = t

    if save_history:
        return state, hist, t_hist, u_hist, Q_hist, g_hist
    return state

# ─────────────────────────────────────────────────────────────────
# ФУНКЦИЯ НЕВЯЗКИ
#
# Изменение логики:
#   • если приземление произошло (landed=True):
#       z = lam2·v² + lam3·(ψ₃+1)² + lam4·H²
#       высота не штрафуется — она гарантированно равна 0
#   • если земля не достигнута за T_final (landed=False):
#       z += lam1·h²  (большой штраф за промах по высоте)
# ─────────────────────────────────────────────────────────────────
def residual(params):
    """
    Оптимизируем только ψ(0) = [ψ₁, ψ₂, ψ₃].
    T_MAX фиксирован — оптимизатор не может «остановить» интегрирование
    в удобный момент. Посадка определяется исключительно детектором h=0.

    • landed=True  → z = lam2·v²(t_land) + лёгкие штрафы на ψ₃, H
    • landed=False → z = 1e6 + lam1·h²   (оптимизатор обязан прийти к земле)
    """
    psi0 = params[:3]
    state = integrate(psi0, T_MAX)

    if not state['landed']:
        return 1e6 + lam1 * state['x1'] ** 2

    H_T = hamiltonian(state)
    z = (lam2 * state['x2'] ** 2
         + lam3 * (state['psi3'] + 1.0) ** 2
         + lam4 * H_T ** 2)
    return z

# ─────────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────
def solve():
    print('\nШаг 1: Глобальный поиск (дифференциальная эволюция)...')

    bounds = [
        (-5.0,  5.0),   # ψ₁(0)
        (-5.0,  5.0),   # ψ₂(0)
        (-3.0,  0.0),   # ψ₃(0)
        # T не оптимизируется — зафиксирован как T_MAX
    ]

    t0 = time.time()
    _ctx  = _mp.get_context('fork')
    _pool = _ctx.Pool()
    res_global = differential_evolution(
        residual,
        bounds,
        maxiter=500,
        tol=1e-10,
        seed=42,
        popsize=30,
        mutation=(0.5, 1.5),
        recombination=0.9,
        updating='deferred',
        workers=_pool.map,
        disp=False,
    )
    _pool.close()
    _pool.join()
    print(f' Завершён за {time.time()-t0:.1f} с')
    print(f' z_global = {res_global.fun:.6e}')
    print(f' ψ(0) = {res_global.x}')

    # ── Шаг 2: least_squares (Левенберг–Марквардт) ─────────────────
    # Решаем систему из 3 уравнений с 3 неизвестными:
    #   r[0] = v(t_land)        → 0   (мягкая посадка)
    #   r[1] = ψ₃(t_land) + 1  → 0   (трансверсальность)
    #   r[2] = H(t_land)        → 0   (трансверсальность по T)
    # Это точное попадание в ноль, без компромиссов весов.
    def vector_residual(psi0):
        state = integrate(psi0, T_MAX)
        if not state['landed']:
            # Если не приземлились — большая невязка по всем компонентам
            return [1e4, 1e4, 1e4]
        return [
            state['x2'],               # v(t_land) → 0
            state['psi3'] + 1.0,       # ψ₃(t_land) + 1 → 0
            hamiltonian(state),         # H(t_land) → 0
        ]

    print('\nШаг 2: Уточнение (least_squares — Левенберг-Марквардт)...')
    t0 = time.time()
    res_local = least_squares(
        vector_residual,
        res_global.x,
        method='lm',           # Левенберг-Марквардт — лучший для гладких систем
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=50_000,
    )
    print(f' Завершён за {time.time()-t0:.1f} с')
    print(f' |r|²     = {(res_local.fun**2).sum():.6e}')
    print(f' r = [v={res_local.fun[0]:.4e}, ψ₃+1={res_local.fun[1]:.4e},'
          f' H={res_local.fun[2]:.4e}]')
    print(f' ψ₁(0) = {res_local.x[0]:.8f}')
    print(f' ψ₂(0) = {res_local.x[1]:.8f}')
    print(f' ψ₃(0) = {res_local.x[2]:.8f}')
    print(f' T_MAX = {T_MAX:.1f} с (фиксирован, фактическое время — из детектора)')
    return res_local

# ─────────────────────────────────────────────────────────────────
# ГРАФИКИ
# ─────────────────────────────────────────────────────────────────
FONT = 'Times New Roman'
FS   = 14
plt.rcParams['font.family']      = FONT
plt.rcParams['axes.titlesize']   = 12
plt.rcParams['axes.labelsize']   = FS
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_results(params_opt):
    psi0_opt = params_opt[:3]

    state, hist, t_hist, u_hist, Q_hist, g_hist = integrate(
        psi0_opt, T_MAX, save_history=True)

    t         = np.array(t_hist)
    T         = state['t']
    fuel_used = m0 - state['x3']
    landed    = state['landed']

    print('\n' + '=' * 70)
    print(' РЕЗУЛЬТАТЫ ОПТИМАЛЬНОГО РЕШЕНИЯ')
    print('=' * 70)
    print(f' Приземление произошло  = {landed}')
    print(f' Время посадки T        = {T:.2f} с  (T_MAX = {T_MAX:.0f} с)')
    print(f' Высота в момент T      = {state["x1"]:.6f} м  (цель: 0)')
    print(f' Скорость в момент T    = {state["x2"]:.6f} м/с (цель: 0)')
    print(f' psi3(T)                = {state["psi3"]:.6f}  (цель: −1)')
    print(f' Конечная масса         = {state["x3"]:.2f} кг')
    print(f' Расход топлива         = {fuel_used:.2f} кг')
    print(f' Остаток топлива        = {state["x3"] - m_dry:.2f} кг')
    print('=' * 70)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title_status = "[OK] Приземление: h=0, v=0" if landed else "[!!] Земля не достигнута"
    fig.suptitle(
        f'Оптимальная посадка с переменной гравитацией g(h)\n'
        f'$T_{{\\rm land}} = {T:.1f}$ с, расход топлива $= {fuel_used:.1f}$ кг — {title_status}',
        fontsize=13)

    ax = axes[0, 0]
    ax.plot(t, hist['x1'], 'b', linewidth=2)
    ax.axhline(0, color='brown', linestyle='--', linewidth=1.5, label='Поверхность')
    if landed:
        ax.plot(T, 0, 'rs', markersize=10, label=f'Касание t={T:.1f}с')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Высота $h$, м')
    ax.set_title('Высота от времени'); ax.legend(fontsize=9); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, hist['x2'], 'g', linewidth=2, label='v(t)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, label='v = 0')
    if landed:
        ax.plot(T, state['x2'], 'rs', markersize=10,
                label=f'касание v={state["x2"]:.3f} м/с')
    ax.set_xlabel('Время, с'); ax.set_ylabel(r'Скорость $\dot{h}$, м/с')
    ax.set_title('Вертикальная скорость'); ax.legend(fontsize=9); ax.grid(True)

    ax = axes[0, 2]
    ax.plot(t, hist['x3'], 'r', linewidth=2)
    ax.axhline(m_dry, color='orange', linestyle='--', linewidth=1.5, label='Сухая масса')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Масса $m$, кг')
    ax.set_title('Масса КА'); ax.legend(fontsize=9); ax.grid(True)

    ax = axes[0, 3]
    ax.plot(t, g_hist, color='purple', linewidth=2)
    ax.axhline(g0, color='gray', linestyle='--', linewidth=1, label=f'$g_0={g0}$ м/с²')
    ax.set_xlabel('Время, с'); ax.set_ylabel('$g(h)$, м/с²')
    ax.set_title('Переменная гравитация'); ax.legend(fontsize=9); ax.grid(True)

    ax = axes[1, 0]
    ax.step(t, u_hist, 'k', linewidth=2, where='post')
    ax.axhline(u_m, color='red', linestyle='--', linewidth=1.5,
               label=f'$u_m={u_m:.3f}$ кг/с')
    ax.set_xlabel('Время, с'); ax.set_ylabel('$u$, кг/с')
    ax.set_title('Управление — расход топлива')
    ax.set_ylim(-0.1 * u_m, 1.2 * u_m); ax.legend(fontsize=9); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, hist['psi1'], label=r'$\psi_1$', linewidth=2)
    ax.plot(t, hist['psi2'], label=r'$\psi_2$', linewidth=2)
    ax.plot(t, hist['psi3'], label=r'$\psi_3$', linewidth=2)
    ax.axhline(-1, color='gray', linestyle=':', linewidth=1.5, label=r'$\psi_3(T)=-1$')
    ax.set_xlabel('Время, с'); ax.set_ylabel(r'$\psi$')
    ax.set_title('Сопряжённые переменные'); ax.legend(fontsize=9); ax.grid(True)

    ax = axes[1, 2]
    ax.plot(t, Q_hist, 'm', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Переключение')
    ax.fill_between(t, Q_hist, 0, where=[q < 0 for q in Q_hist],
                    alpha=0.2, color='red', label='Двигатель ВКЛ')
    ax.fill_between(t, Q_hist, 0, where=[q >= 0 for q in Q_hist],
                    alpha=0.2, color='blue', label='Двигатель ВЫКЛ')
    ax.set_xlabel('Время, с'); ax.set_ylabel(r'$Q$')
    ax.set_title('Функция переключения'); ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1, 3]
    sc = ax.scatter(hist['x2'], hist['x1'], c=t, cmap='viridis', s=8)
    plt.colorbar(sc, ax=ax, label='Время, с')
    ax.plot(hist['x2'][0],  hist['x1'][0],  'go', markersize=10, label='Начало')
    ax.plot(hist['x2'][-1], hist['x1'][-1], 'rs', markersize=10, label='Посадка')
    ax.set_xlabel(r'Скорость, м/с'); ax.set_ylabel('Высота, м')
    ax.set_title('Фазовый портрет'); ax.legend(fontsize=9); ax.grid(True)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ПРОГРАММА
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_total = time.time()
    res = solve()
    print(f'\nОбщее время: {time.time() - t_total:.1f} с')
    print('\nПостроение графиков...')
    plot_results(res.x)