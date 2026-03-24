"""
=======================================================================
УЛУЧШЕННАЯ ВЕРСИЯ — Оптимальная посадка на Луну

Ключевое отличие от предыдущих версий:
- Время посадки T включено в число искомых переменных (4-й параметр).
  Это убирает нестабильность event-детектора и гладко соединяет
  все три терминальных условия в одну функцию невязки.
=======================================================================
"""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# ИСХОДНЫЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────
R_moon = 1_737_000.0   # радиус Луны, м
g0     = 1.62          # ускорение на поверхности, м/с²

h0     = 15_000.0      # начальная высота, м
dh0    = -20.0         # начальная вертикальная скорость, м/с
m0     = 5_560.0       # начальная масса, кг
m_dry  = 3_740.0       # сухая масса (без топлива), кг
mf     = m0 - m_dry    # запас топлива, кг
c      = 3_050.0       # удельный импульс, м/с
P_max  = 20_000.0      # максимальная тяга, Н
u_m    = P_max / c     # максимальный расход, кг/с

X0 = [h0, dh0, m0]

# Нормированные веса: все слагаемые невязки одного порядка
lam1 = 1.0 / h0**2          # x₁ ~ h0
lam2 = 1.0 / abs(dh0)**2    # x₂ ~ dh0
lam3 = 1.0                  # ψ₃ безразмерна

DT      = 1.0    # базовый шаг РК4, с
DT_FINE = 0.1    # шаг вблизи поверхности (h < 500 м)

print('=' * 70)
print(' УЛУЧШЕННАЯ ВЕРСИЯ: Оптимальная посадка на Луну (Н1-Л3)')
print('=' * 70)
print(f' h₀     = {h0} м')
print(f' ẋ₂(0)  = {dh0} м/с')
print(f' m₀     = {m0} кг')
print(f' m_dry  = {m_dry} кг  (запас топлива = {mf:.0f} кг)')
print(f' c      = {c} м/с')
print(f' P_max  = {P_max} Н  →  u_m = {u_m:.4f} кг/с')
print(f' g₀     = {g0} м/с²')
print(f' R_moon = {R_moon/1000:.0f} км')
g_at_h0 = g0 * (R_moon / (R_moon + h0)) ** 2
print(f' g(h₀)  = {g_at_h0:.4f} м/с²  (Δg = {(g0-g_at_h0)/g0*100:.3f}%)')
print('=' * 70)

# ─────────────────────────────────────────────────────────────────
# ПЕРЕМЕННАЯ ГРАВИТАЦИЯ
# ─────────────────────────────────────────────────────────────────
def g_func(h):
    return g0 * (R_moon / (R_moon + max(h, 0.0))) ** 2

def dg_dh(h):
    return -2.0 * g0 * R_moon**2 / (R_moon + max(h, 0.0))**3

# ─────────────────────────────────────────────────────────────────
# ФУНКЦИЯ РК4
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
    """Функция переключения Q = ψ₂·c/x₃ − ψ₃"""
    return state['psi2'] * c / state['x3'] - state['psi3']

def control(state):
    if state['x3'] <= m_dry:
        return 0.0
    return 0.0 if Q_switch(state) >= 0 else u_m

# ─────────────────────────────────────────────────────────────────
# ПРАВЫЕ ЧАСТИ КАНОНИЧЕСКОЙ СИСТЕМЫ
# ─────────────────────────────────────────────────────────────────
def dx1_eq(state):
    return state['x2'], 'x1'

def dx2_eq(state):
    u = control(state)
    return c * u / state['x3'] - g_func(state['x1']), 'x2'

def dx3_eq(state):
    return -control(state), 'x3'

def dpsi1_eq(state):
    # ψ̇₁ = −∂H/∂x₁ = ψ₂·dg/dh  (ненулевое при переменном g(h))
    return state['psi2'] * dg_dh(state['x1']), 'psi1'

def dpsi2_eq(state):
    return -state['psi1'], 'psi2'

def dpsi3_eq(state):
    u = control(state)
    return c * u * state['psi2'] / (state['x3'] ** 2), 'psi3'

EQUATIONS = [dx1_eq, dx2_eq, dx3_eq, dpsi1_eq, dpsi2_eq, dpsi3_eq]
DX_KEYS   = ['x1', 'x2', 'x3', 'psi1', 'psi2', 'psi3']

# ─────────────────────────────────────────────────────────────────
# ИНТЕГРИРОВАНИЕ НА ФИКСИРОВАННОЕ ВРЕМЯ T_final
# (T_final — 4-й оптимизируемый параметр, event-детектор не нужен)
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
        dt = min(dt, T_final - t)   # не перешагнуть через T_final

        vals = runge_kutta_4(EQUATIONS, state, dt, DX_KEYS)

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

    state['t'] = T_final

    if save_history:
        return state, hist, t_hist, u_hist, Q_hist, g_hist
    return state

# ─────────────────────────────────────────────────────────────────
# ФУНКЦИЯ НЕВЯЗКИ  (params = [ψ₁(0), ψ₂(0), ψ₃(0), T])
# ─────────────────────────────────────────────────────────────────
def residual(params):
    """
    z = λ₁·x₁²(T) + λ₂·x₂²(T) + λ₃·(ψ₃(T)+1)²
    Оптимизируется по [ψ₁(0), ψ₂(0), ψ₃(0), T].
    При z→0: x₁(T)→0 (касание), x₂(T)→0 (мягкая посадка), ψ₃(T)→−1.
    """
    psi0   = params[:3]
    T_fin  = params[3]
    if T_fin <= 0:
        return 1e10
    state = integrate(psi0, T_fin)
    z = (lam1 * state['x1'] ** 2 +
         lam2 * state['x2'] ** 2 +
         lam3 * (state['psi3'] + 1.0) ** 2)
    return z

# ─────────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────
def solve():
    print('\nШаг 1: Глобальный поиск (дифференциальная эволюция)...')

    bounds = [
        (-1.0,  1.0),    # ψ₁(0)
        (-1.0,  0.0),    # ψ₂(0)
        (-3.0,  0.0),    # ψ₃(0)
        (100.0, 2000.0), # T, с  — время посадки (4-й параметр)
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
    print(f' Завершён за {time.time()-t0:.1f} с')
    print(f' z_global = {res_global.fun:.6e}')
    print(f' ψ(0) = {res_global.x[:3]},  T = {res_global.x[3]:.1f} с')

    print('\nШаг 2: Уточнение (Нелдер–Мид)...')
    t0 = time.time()
    res_local = minimize(
        residual,
        res_global.x,
        method='Nelder-Mead',
        options={
            'xatol':    1e-12,
            'fatol':    1e-14,
            'maxiter':  200_000,
            'adaptive': True,
        },
    )
    print(f' Завершён за {time.time()-t0:.1f} с')
    print(f' z_final  = {res_local.fun:.6e}')
    print(f' ψ₁(0) = {res_local.x[0]:.8f}')
    print(f' ψ₂(0) = {res_local.x[1]:.8f}')
    print(f' ψ₃(0) = {res_local.x[2]:.8f}')
    print(f' T     = {res_local.x[3]:.4f} с')
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
    T_opt    = params_opt[3]

    state, hist, t_hist, u_hist, Q_hist, g_hist = integrate(
        psi0_opt, T_opt, save_history=True)

    t         = np.array(t_hist)
    T         = state['t']
    fuel_used = m0 - state['x3']

    print('\n' + '=' * 70)
    print(' РЕЗУЛЬТАТЫ ОПТИМАЛЬНОГО РЕШЕНИЯ')
    print('=' * 70)
    print(f' Время посадки T        = {T:.2f} с')
    print(f' Высота в момент T      = {state["x1"]:.4f} м  (цель: 0)')
    print(f' Скорость в момент T    = {state["x2"]:.4f} м/с (цель: 0)')
    print(f' psi3(T)                = {state["psi3"]:.6f}  (цель: −1)')
    print(f' Конечная масса         = {state["x3"]:.2f} кг')
    print(f' Расход топлива         = {fuel_used:.2f} кг')
    print(f' Остаток топлива        = {state["x3"] - m_dry:.2f} кг')
    print('=' * 70)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.canvas.manager.set_window_title(
        'Оптимальная посадка — Н1-Л3 (улучшенная версия)')
    fig.suptitle(
        f'Оптимальная посадка с переменной гравитацией g(h)\n'
        f'$T = {T:.1f}$ с, расход топлива $= {fuel_used:.1f}$ кг',
        fontsize=14)

    # ── 1. Высота ──
    ax = axes[0, 0]
    ax.plot(t, hist['x1'], 'b', linewidth=2)
    ax.axhline(0, color='brown', linestyle='--', linewidth=1, label='Поверхность')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Высота $h$, м')
    ax.set_title('Высота от времени'); ax.legend(fontsize=10); ax.grid(True)

    # ── 2. Скорость ──
    ax = axes[0, 1]
    ax.plot(t, hist['x2'], 'g', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Время, с'); ax.set_ylabel(r'Скорость $\dot{h}$, м/с')
    ax.set_title('Вертикальная скорость'); ax.grid(True)

    # ── 3. Масса ──
    ax = axes[0, 2]
    ax.plot(t, hist['x3'], 'r', linewidth=2)
    ax.axhline(m_dry, color='orange', linestyle='--', linewidth=1.5, label='Сухая масса')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Масса $m$, кг')
    ax.set_title('Масса КА'); ax.legend(fontsize=10); ax.grid(True)

    # ── 4. Гравитация g(h) ──
    ax = axes[0, 3]
    ax.plot(t, g_hist, color='purple', linewidth=2)
    ax.axhline(g0, color='gray', linestyle='--', linewidth=1, label=f'$g_0={g0}$ м/с²')
    ax.set_xlabel('Время, с'); ax.set_ylabel('$g(h)$, м/с²')
    ax.set_title('Переменная гравитация'); ax.legend(fontsize=10); ax.grid(True)

    # ── 5. Управление u(t) ──
    ax = axes[1, 0]
    ax.step(t, u_hist, 'k', linewidth=2, where='post')
    ax.axhline(u_m, color='red', linestyle='--', linewidth=1.5, label=f'$u_m={u_m:.3f}$ кг/с')
    ax.set_xlabel('Время, с'); ax.set_ylabel('$u$, кг/с')
    ax.set_title(r'Управление — расход топлива')
    ax.set_ylim(-0.1 * u_m, 1.2 * u_m); ax.legend(fontsize=10); ax.grid(True)

    # ── 6. Сопряжённые переменные ──
    ax = axes[1, 1]
    ax.plot(t, hist['psi1'], label=r'$\psi_1$', linewidth=2)
    ax.plot(t, hist['psi2'], label=r'$\psi_2$', linewidth=2)
    ax.plot(t, hist['psi3'], label=r'$\psi_3$', linewidth=2)
    ax.axhline(-1, color='gray', linestyle=':', linewidth=1.5, label=r'$\psi_3(T)=-1$')
    ax.set_xlabel('Время, с'); ax.set_ylabel(r'$\psi$')
    ax.set_title(r'Сопряжённые переменные'); ax.legend(fontsize=10); ax.grid(True)

    # ── 7. Функция переключения Q(t) ──
    ax = axes[1, 2]
    ax.plot(t, Q_hist, 'm', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Переключение')
    ax.fill_between(t, Q_hist, 0, where=[q < 0 for q in Q_hist],
                    alpha=0.2, color='red', label='Двигатель ВКЛ')
    ax.fill_between(t, Q_hist, 0, where=[q >= 0 for q in Q_hist],
                    alpha=0.2, color='blue', label='Двигатель ВЫКЛ')
    ax.set_xlabel('Время, с'); ax.set_ylabel(r'$Q$')
    ax.set_title(r'Функция переключения'); ax.legend(fontsize=9); ax.grid(True)

    # ── 8. Фазовый портрет ──
    ax = axes[1, 3]
    sc = ax.scatter(hist['x2'], hist['x1'], c=t, cmap='viridis', s=8)
    plt.colorbar(sc, ax=ax, label='Время, с')
    ax.plot(hist['x2'][0],  hist['x1'][0],  'go', markersize=10, label='Начало')
    ax.plot(hist['x2'][-1], hist['x1'][-1], 'rs', markersize=10, label='Посадка')
    ax.set_xlabel(r'Скорость, м/с'); ax.set_ylabel('Высота, м')
    ax.set_title(r'Фазовый портрет'); ax.legend(fontsize=10); ax.grid(True)

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
