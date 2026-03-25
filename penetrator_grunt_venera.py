"""
=====================================================================
 МОДЕЛИРОВАНИЕ ВНЕДРЕНИЯ ЗОНДА-ПЕНЕТРАТОРА В ГРУНТ ВЕНЕРЫ
 Модель Понселе | шаг Qk = 0.5°, шаг V0 = 1 м/с
 Параллелизм: pool.starmap — без Pipe, без ограничений на macOS
=====================================================================
"""

import math as m
import time
import multiprocessing
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
#  ПАРАМЕТРЫ ПЕНЕТРАТОРА
# ─────────────────────────────────────────────────────────────────
MASS    = 120.0
D_BODY  = 0.6
S_M     = m.pi * (D_BODY / 2.0) ** 2
G_VENUS = 8.87
G0      = 9.81   # ← стандартное g для нормировки перегрузки, м/с²

# ─────────────────────────────────────────────────────────────────
#  СВОЙСТВА БАЗОВЫХ ГРУНТОВ
# ─────────────────────────────────────────────────────────────────
SEDIMENT = {'sigma_c': 3e6,  'rho_s': 1500.0, 'label': 'Осадок'}
BASALT   = {'sigma_c': 80e6, 'rho_s': 2800.0, 'label': 'Базальт'}

# ─────────────────────────────────────────────────────────────────
#  ОДНОРОДНЫЕ ГРУНТЫ
# ─────────────────────────────────────────────────────────────────
PURE_SOILS = {
    'Осадочный грунт (Венера-13,  σ_c = 3 МПа)': SEDIMENT,
    'Базальт (Венера-14,  σ_c = 80 МПа)':         BASALT,
}

# ─────────────────────────────────────────────────────────────────
#  СЛОИСТЫЕ ГРУНТЫ  (layers циклически повторяются)
# ─────────────────────────────────────────────────────────────────
MIXED_SOILS = {
    'Слоистый: осадок 0.5 м / базальт 0.5 м': {
        'layers': [(SEDIMENT, 0.5), (BASALT, 0.5)]
    },
    'Слоистый: осадок 1 м / базальт 0.5 м': {
        'layers': [(SEDIMENT, 1.0), (BASALT, 0.5)]
    },
    'Слоистый: осадок 2 м / базальт 1 м': {
        'layers': [(SEDIMENT, 2.0), (BASALT, 1.0)]
    },
    'Слоистый: осадок 0.5 м / базальт 2 м': {
        'layers': [(SEDIMENT, 0.5), (BASALT, 2.0)]
    },
}

# ─────────────────────────────────────────────────────────────────
#  СЕТКА ПАРАМЕТРОВ
# ─────────────────────────────────────────────────────────────────
QK_STEP = 0.5
V0_STEP = 1

QK_VALUES = [round(10.0 + i * QK_STEP, 2)
             for i in range(int((90 - 10) / QK_STEP) + 1)]   # 161 значение
V0_VALUES = list(range(50, 136, V0_STEP))                     # 86 значений

X_REQUIRED = 3.0     # требуемая глубина внедрения, м
X_MAX_SIM  = 30.0    # ограничение симуляции
DT         = 1e-4    # шаг РК4, с

cToRad = m.pi / 180.0


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
#  УРАВНЕНИЯ ДВИЖЕНИЯ В ГРУНТЕ
# ─────────────────────────────────────────────────────────────────
def dV_grunt(state):
    return G_VENUS - S_M * (state['A'] + state['B'] * state['V'] ** 2) / MASS, 'V'


def dx_grunt(state):
    return state['V'], 'x'


EQUATIONS_GRUNT = [dV_grunt, dx_grunt]
DX_KEYS_GRUNT   = ['V', 'x']


# ─────────────────────────────────────────────────────────────────
#  КОЭФФИЦИЕНТЫ ПОНСЕЛЕ
# ─────────────────────────────────────────────────────────────────
def poncelet_coeffs(Qk_deg, sigma_c, rho_s):
    alpha = (Qk_deg / 2.0) * cToRad
    sin2  = m.sin(alpha) ** 2
    cos2  = m.cos(alpha) ** 2
    A     = sigma_c * sin2
    B     = rho_s * sin2 * (1.0 + cos2) / 2.0
    return A, B


# ─────────────────────────────────────────────────────────────────
#  ВОРКЕРЫ — возвращают результат напрямую (pool.starmap)
# ─────────────────────────────────────────────────────────────────
def worker_pure(V0, Qk_deg, sigma_c, rho_s):
    """
    Однородный грунт.
    Возвращает (V0, Qk_deg, x_max, V_hist, x_hist, max_g).
    """
    A, B    = poncelet_coeffs(Qk_deg, sigma_c, rho_s)
    state   = {'V': float(V0), 'x': 0.0, 'A': A, 'B': B}
    V_hist  = [float(V0)]
    x_hist  = [0.0]
    step    = 0
    max_g   = S_M * (A + B * float(V0) ** 2) / (MASS * G0)  # ← начальная перегрузка

    while True:
        vals  = runge_kutta_4(EQUATIONS_GRUNT, state, DT, DX_KEYS_GRUNT)
        V_new = vals[0]
        x_new = vals[1]
        if V_new <= 0.0 or x_new >= X_MAX_SIM:
            break
        g_step = S_M * (A + B * V_new ** 2) / (MASS * G0)   # ← перегрузка на шаге
        if g_step > max_g:                                    # ← фиксация максимума
            max_g = g_step                                    # ←
        state['V'] = V_new
        state['x'] = x_new
        step += 1
        if step % 10 == 0:
            V_hist.append(V_new)
            x_hist.append(x_new)

    return V0, Qk_deg, state['x'], V_hist, x_hist, max_g     # ← добавлен max_g


def worker_mixed(V0, Qk_deg, layers_precomp):
    """
    Слоистый грунт.
    layers_precomp: tuple of (A, B, thickness)
    Возвращает (V0, Qk_deg, x_max, V_hist, x_hist, max_g).
    """
    total  = sum(t for _, _, t in layers_precomp)
    x_cur  = 0.0
    V_cur  = float(V0)
    V_hist = [V_cur]
    x_hist = [x_cur]
    step   = 0
    max_g  = 0.0                                                   # ←

    while True:
        # определяем слой по текущей глубине
        x_mod = x_cur % total
        acc   = 0.0
        A_cur, B_cur = layers_precomp[-1][0], layers_precomp[-1][1]
        for A_l, B_l, thick in layers_precomp:
            acc += thick
            if x_mod < acc:
                A_cur, B_cur = A_l, B_l
                break

        state = {'V': V_cur, 'x': x_cur, 'A': A_cur, 'B': B_cur}
        vals  = runge_kutta_4(EQUATIONS_GRUNT, state, DT, DX_KEYS_GRUNT)
        V_new = vals[0]
        x_new = vals[1]
        if V_new <= 0.0 or x_new >= X_MAX_SIM:
            break
        g_step = S_M * (A_cur + B_cur * V_new ** 2) / (MASS * G0)  # ← перегрузка
        if g_step > max_g:                                           # ← фиксация
            max_g = g_step                                           # ←
        V_cur = V_new
        x_cur = x_new
        step += 1
        if step % 10 == 0:
            V_hist.append(V_cur)
            x_hist.append(x_cur)

    return V0, Qk_deg, x_cur, V_hist, x_hist, max_g                # ← добавлен max_g


# ─────────────────────────────────────────────────────────────────
#  ЗАПУСК РАСЧЁТА + СБОРКА СЕТКИ
# ─────────────────────────────────────────────────────────────────
def build_grid(raw_results):
    """raw_results: list of (V0, Qk_deg, x_max, V_hist, x_hist, max_g)"""
    n_V0 = len(V0_VALUES)
    n_Qk = len(QK_VALUES)
    results_grid = np.zeros((n_V0, n_Qk))
    max_g_grid   = np.zeros((n_V0, n_Qk))   # ←
    trajectories = {}

    for V0_r, Qk_r, x_max, V_hist, x_hist, max_g in raw_results:   # ← распаковка
        i = V0_VALUES.index(int(round(V0_r)))
        j = min(range(n_Qk), key=lambda k: abs(QK_VALUES[k] - Qk_r))
        results_grid[i, j] = x_max
        max_g_grid[i, j]   = max_g                                  # ←
        trajectories[(int(V0_r), round(Qk_r, 1))] = (V_hist, x_hist)

    return results_grid, max_g_grid, trajectories                   # ←


def run_grid_pure(sigma_c, rho_s, n_cpu):
    args = [(V0, Qk, sigma_c, rho_s)
            for V0 in V0_VALUES for Qk in QK_VALUES]
    t0   = time.time()
    with multiprocessing.Pool(processes=n_cpu) as pool:
        raw = pool.starmap(worker_pure, args)
    print(f'  Расчёт завершён за {time.time() - t0:.1f} с')
    return build_grid(raw)


def run_grid_mixed(layers, n_cpu):
    args = []
    for V0 in V0_VALUES:
        for Qk in QK_VALUES:
            lp = tuple(
                (poncelet_coeffs(Qk, s['sigma_c'], s['rho_s'])[0],
                 poncelet_coeffs(Qk, s['sigma_c'], s['rho_s'])[1],
                 thick)
                for s, thick in layers
            )
            args.append((V0, Qk, lp))
    t0 = time.time()
    with multiprocessing.Pool(processes=n_cpu) as pool:
        raw = pool.starmap(worker_mixed, args)
    print(f'  Расчёт завершён за {time.time() - t0:.1f} с')
    return build_grid(raw)


# ─────────────────────────────────────────────────────────────────
#  ГРАФИКИ
# ─────────────────────────────────────────────────────────────────
FONT = 'Times New Roman'
FS   = 14
plt.rcParams['font.family']    = FONT
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = FS


def plot_soil(soil_name, results_grid, max_g_grid, trajectories, layers=None):  # ← +max_g_grid
    V0a = np.array(V0_VALUES, dtype=float)
    Qka = np.array(QK_VALUES,  dtype=float)
    tag = soil_name[:28]

    # ── 1. Тепловая карта ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title(f'[{tag}] 1 — Тепловая карта')
    vmax = min(results_grid.max(), X_MAX_SIM)
    pcm  = ax.pcolormesh(Qka, V0a, results_grid,
                         cmap='jet', shading='auto', vmin=0, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label='Глубина внедрения, м')
    if results_grid.max() >= X_REQUIRED:
        cs = ax.contour(Qka, V0a, results_grid,
                        levels=[X_REQUIRED], colors='white', linewidths=2.5)
        ax.clabel(cs, fmt=f'x = {X_REQUIRED} м', fontsize=11, colors='white')
    ax.set_xlabel('Угол раствора конуса, °')
    ax.set_ylabel('Скорость внедрения, м/с')
    ax.set_title(f'Глубина внедрения зонда-пенетратора\n{soil_name}')
    ax.grid(alpha=0.2, color='white')
    plt.tight_layout()

    # ── 2. Карта надёжности ───────────────────────────────────────
    reliability = (results_grid >= X_REQUIRED).astype(float)
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title(f'[{tag}] 2 — Надёжность')
    pcm2 = ax.pcolormesh(Qka, V0a, reliability,
                         cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
    cb2  = fig.colorbar(pcm2, ax=ax, ticks=[0.0, 0.5, 1.0])
    cb2.ax.set_yticklabels(['Недостаточно', '', 'Выполнено'], fontsize=11)
    if 0 < reliability.max() and reliability.min() < 1:
        ax.contour(Qka, V0a, reliability,
                   levels=[0.5], colors='black', linewidths=2.5, linestyles='--')
    ax.set_xlabel('Угол раствора конуса, °')
    ax.set_ylabel('Скорость внедрения, м/с')
    ax.set_title(f'Критерий надёжности (x ≥ {X_REQUIRED} м)\n{soil_name}')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # ── 3. Тепловая карта максимальной перегрузки ─────────────── # ←
    fig, ax = plt.subplots(figsize=(11, 6))                        # ←
    fig.canvas.manager.set_window_title(f'[{tag}] 3 — Макс. перегрузка')  # ←
    pcm3 = ax.pcolormesh(Qka, V0a, max_g_grid,                    # ←
                         cmap='hot_r', shading='auto')             # ←
    fig.colorbar(pcm3, ax=ax, label='Максимальная перегрузка, g') # ←
    ax.set_xlabel('Угол раствора конуса, °')                      # ←
    ax.set_ylabel('Скорость внедрения, м/с')                      # ←
    ax.set_title(f'Макс. перегрузка (пик: {max_g_grid.max():.0f} g)\n{soil_name}')  # ←
    ax.grid(alpha=0.2, color='white')                              # ←
    plt.tight_layout()                                             # ←

    # ── 4. Глубина(V0) при фиксированных углах ───────────────────
    sel_qk  = [10, 20, 30, 45, 60, 75, 90]
    colors7 = plt.cm.tab10(np.linspace(0, 0.6, len(sel_qk)))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title(f'[{tag}] 4 — Глубина vs V0')
    for col, Qk in zip(colors7, sel_qk):
        j = min(range(len(QK_VALUES)), key=lambda k: abs(QK_VALUES[k] - Qk))
        ax.plot(V0a, results_grid[:, j],
                label=f'Qk = {QK_VALUES[j]}°', color=col, linewidth=2)
    ax.axhline(y=X_REQUIRED, color='red', linestyle='--',
               linewidth=1.8, label=f'x_треб = {X_REQUIRED} м')
    ax.set_xlabel('Скорость внедрения, м/с')
    ax.set_ylabel('Глубина внедрения, м')
    ax.set_title(f'Зависимость глубины от скорости\n{soil_name}')
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()

    # ── 5. Глубина(Qk) при фиксированных скоростях ───────────────
    sel_v0  = [50, 65, 80, 100, 120, 135]
    colors6 = plt.cm.tab10(np.linspace(0, 0.5, len(sel_v0)))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title(f'[{tag}] 5 — Глубина vs Qk')
    for col, V0 in zip(colors6, sel_v0):
        i = min(range(len(V0_VALUES)), key=lambda k: abs(V0_VALUES[k] - V0))
        ax.plot(Qka, results_grid[i, :],
                label=f'V0 = {V0_VALUES[i]} м/с', color=col, linewidth=2)
    ax.axhline(y=X_REQUIRED, color='red', linestyle='--',
               linewidth=1.8, label=f'x_треб = {X_REQUIRED} м')
    ax.set_xlabel('Угол раствора конуса, °')
    ax.set_ylabel('Глубина внедрения, м')
    ax.set_title(f'Зависимость глубины от угла конуса\n{soil_name}')
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()

    # ── 6. Траектории V(x) в грунте ──────────────────────────────
    cases   = [(135, 90.0), (135, 45.0), (100, 45.0), (75, 30.0), (50, 15.0)]
    colors5 = plt.cm.tab10(np.linspace(0, 0.45, len(cases)))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title(f'[{tag}] 6 — V(x) в грунте')

    if layers is not None:
        layer_colors  = ['#ffe8c0', '#b0b0b0']
        drawn_labels  = set()
        x_acc         = 0.0
        total         = sum(t for _, t in layers)
        repeats       = int(X_MAX_SIM / total) + 2
        for _ in range(repeats):
            for li, (soil, thick) in enumerate(layers):
                lbl = soil['label'] if soil['label'] not in drawn_labels else ''
                ax.axvspan(x_acc, x_acc + thick,
                           alpha=0.25, color=layer_colors[li % 2], label=lbl)
                if lbl:
                    drawn_labels.add(soil['label'])
                x_acc += thick
                if x_acc >= X_MAX_SIM:
                    break
            if x_acc >= X_MAX_SIM:
                break

    for col, (V0, Qk) in zip(colors5, cases):
        key = (V0, Qk)
        if key not in trajectories:
            continue
        V_h, x_h = trajectories[key]
        ax.plot(x_h, V_h, label=f'V0={V0} м/с, Qk={Qk}°',
                color=col, linewidth=2)
    ax.axvline(x=X_REQUIRED, color='red', linestyle='--',
               linewidth=1.8, label=f'x_треб = {X_REQUIRED} м')
    ax.set_xlabel('Глубина внедрения, м')
    ax.set_ylabel('Скорость пенетратора, м/с')
    ax.set_title(f'Изменение скорости при движении в грунте\n{soil_name}')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    # ── 7. V_min(Qk) ─────────────────────────────────────────────
    V_min_arr = []
    for j in range(len(QK_VALUES)):
        found = False
        for i in range(len(V0_VALUES)):
            if results_grid[i, j] >= X_REQUIRED:
                V_min_arr.append(float(V0_VALUES[i]))
                found = True
                break
        if not found:
            V_min_arr.append(float('nan'))

    V_min_np   = np.array(V_min_arr)
    valid_mask = ~np.isnan(V_min_np)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title(f'[{tag}] 7 — V_min(Qk)')
    ax.plot(Qka, V_min_arr, '-', color='navy', linewidth=2,
            label=f'V_min для x ≥ {X_REQUIRED} м')
    ax.fill_between(Qka,
                    np.where(valid_mask, V_min_np, 136), 136,
                    alpha=0.15, color='red',   label='Внедрение невозможно')
    ax.fill_between(Qka, 50,
                    np.where(valid_mask & (V_min_np > 50), V_min_np, 50),
                    alpha=0.15, color='green', label='Внедрение обеспечено')
    ax.set_xlim(10, 90)
    ax.set_ylim(45, 140)
    ax.set_xlabel('Угол раствора конуса, °')
    ax.set_ylabel('Минимальная скорость внедрения, м/с')
    ax.set_title(f'Минимальная скорость для x ≥ {X_REQUIRED} м\n{soil_name}')
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()


def plot_mixed_comparison(mixed_results):
    n   = len(mixed_results)
    nc  = 2
    nr  = (n + 1) // 2
    V0a = np.array(V0_VALUES, dtype=float)
    Qka = np.array(QK_VALUES,  dtype=float)

    # сводные тепловые карты
    fig, axes = plt.subplots(nr, nc, figsize=(18, 6 * nr))
    fig.canvas.manager.set_window_title('Сравнение слоистых — тепловые карты')
    axes = axes.flatten()
    vmax_global = min(max(rg.max() for rg, _ in mixed_results.values()), X_MAX_SIM)  # ←

    for ax, (sname, (rg, mg)) in zip(axes, mixed_results.items()):   # ←
        pcm = ax.pcolormesh(Qka, V0a, rg,
                            cmap='jet', shading='auto',
                            vmin=0, vmax=vmax_global)
        fig.colorbar(pcm, ax=ax, label='Глубина, м')
        if rg.max() >= X_REQUIRED:
            cs = ax.contour(Qka, V0a, rg,
                            levels=[X_REQUIRED], colors='white', linewidths=2)
            ax.clabel(cs, fmt=f'{X_REQUIRED} м', fontsize=10, colors='white')
        ax.set_xlabel('Угол конуса, °')
        ax.set_ylabel('V₀, м/с')
        ax.set_title(sname, fontsize=11)
        ax.grid(alpha=0.2, color='white')
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(
        f'Слоистые грунты Венеры — глубина внедрения  (x_треб = {X_REQUIRED} м)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    # сравнительный V_min(Qk)
    fig2, ax2 = plt.subplots(figsize=(13, 7))
    fig2.canvas.manager.set_window_title('Сравнение слоистых — V_min(Qk)')
    colors_mix = plt.cm.Set2(np.linspace(0, 1, n))

    for col, (sname, (rg, mg)) in zip(colors_mix, mixed_results.items()):  # ←
        V_min_arr = []
        for j in range(len(QK_VALUES)):
            found = False
            for i in range(len(V0_VALUES)):
                if rg[i, j] >= X_REQUIRED:
                    V_min_arr.append(float(V0_VALUES[i]))
                    found = True
                    break
            if not found:
                V_min_arr.append(float('nan'))
        ax2.plot(Qka, V_min_arr, '-', linewidth=2, color=col, label=sname)

    ax2.axhline(y=50,  color='black', linestyle=':',  linewidth=1,
                label='V₀_min = 50 м/с')
    ax2.axhline(y=135, color='black', linestyle='--', linewidth=1,
                label='V₀_max = 135 м/с')
    ax2.set_xlim(10, 90)
    ax2.set_ylim(45, 145)
    ax2.set_xlabel('Угол раствора конуса, °')
    ax2.set_ylabel('Минимальная скорость внедрения, м/с')
    ax2.set_title(
        f'Сравнение V_min(Qk) для слоистых грунтов  (x_треб = {X_REQUIRED} м)')
    ax2.legend(fontsize=10)
    ax2.grid(True)
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────
#  ГЛАВНАЯ ПРОГРАММА
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    n_cpu = multiprocessing.cpu_count()
    n_V0  = len(V0_VALUES)
    n_Qk  = len(QK_VALUES)

    print(f'Доступно процессоров : {n_cpu}')
    print(f'Размер сетки         : {n_V0} x {n_Qk} = {n_V0 * n_Qk} вариантов')
    print(f'V0  = {V0_VALUES[0]}..{V0_VALUES[-1]} м/с  (шаг {V0_STEP} м/с)')
    print(f'Qk  = {QK_VALUES[0]}..{QK_VALUES[-1]} °    (шаг {QK_STEP} °)')
    print(f'Требуемая глубина    : {X_REQUIRED} м\n')

    total_start = time.time()

    # ── ОДНОРОДНЫЕ ГРУНТЫ ─────────────────────────────────────────
    for soil_name, props in PURE_SOILS.items():
        print('=' * 62)
        print(f'Грунт : {soil_name}')
        print('=' * 62)
        rg, mg, traj = run_grid_pure(props['sigma_c'], props['rho_s'], n_cpu)  # ←
        print(f'  Макс. перегрузка по сетке: {mg.max():.1f} g')                # ←
        plot_soil(soil_name, rg, mg, traj, layers=None)                        # ←

    # ── СЛОИСТЫЕ ГРУНТЫ ───────────────────────────────────────────
    mixed_results = {}
    for soil_name, props in MIXED_SOILS.items():
        print('=' * 62)
        layers    = props['layers']
        layer_str = ' / '.join(f"{s['label']} {t} м" for s, t in layers)
        print(f'Слоистый : {layer_str}  (циклически)')
        print('=' * 62)
        rg, mg, traj = run_grid_mixed(layers, n_cpu)                           # ←
        print(f'  Макс. перегрузка по сетке: {mg.max():.1f} g')               # ←
        mixed_results[soil_name] = (rg, mg)                                    # ←
        plot_soil(soil_name, rg, mg, traj, layers=layers)                      # ←

    plot_mixed_comparison(mixed_results)

    print(f'\nОбщее время: {time.time() - total_start:.1f} с')
    plt.show()