"""
Возмущения НУ по Монте-Карло + стохастический ветер.

Что изменилось относительно исходной версии:
  * ветер — реализация случайной функции высоты (venus_wind.py):
    формирующий фильтр 2-го порядка по высоте + закон Вейбулла для зональной скорости;
  * пространственная (3D) модель движения: широта, долгота, высота;
    на выходе дальность L, высота H и боковое отклонение Z;
  * ветер действует через скорость относительно воздуха (descent_core.rhs);
  * азимут входа выставляется ПО ВЕТРУ в верхних слоях (AZIMUTH_MODE);
  * исправлен РК4 (в исходнике к переменным состояния прибавлялось dt/2);
  * воспроизводимость: все случайные величины — от одного BASE_SEED;
  * пул процессов через pool.map: ошибка в рабочем процессе больше не «вешает» программу.

Нужны файлы venus_wind.py и descent_core.py в той же папке.
"""

import math as m
import multiprocessing
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from venus_wind import VenusWindModel, downwind_azimuth
from descent_core import (DescentParams, simulate, newton_interpolation, cToRad, cToDeg,
                          plot_standard_set, plot_landing, landing_stats)

# ---------------------------------------------------------------------------
# Параметры расчёта
# ---------------------------------------------------------------------------
N_RUNS = 20                 # количество реализаций
BASE_SEED = 2026            # одно число -> весь набор реализаций воспроизводим
DT = 0.01
SAVE_EVERY = 10             # сохранять каждый 10-й шаг (0.1 с)

mass = 120.0
d = 0.8
S = m.pi * d ** 2 / 4

V0_RANGE = (10_900.0, 11_100.0)
TETTA0_RANGE_DEG = (-25.0, -15.0)

# Азимут входа ψ0 (от севера по часовой):
#   "downwind"             — по направлению среднего (климатического) ветра в слое H_ALIGN;
#                            так выставляют вход на практике: реальная реализация ветра заранее неизвестна;
#   "downwind_realization" — по среднему ветру ДАННОЙ реализации в слое H_ALIGN
#                            (идеальное знание ветра — верхняя оценка эффекта);
#   "fixed"                — AZIMUTH_FIXED_DEG (90° — на восток, против суперротации).
AZIMUTH_MODE = "downwind"
AZIMUTH_FIXED_DEG = 90.0
H_ALIGN = (60_000.0, 100_000.0)   # «верхние слои», по которым выставляется азимут, м
AZIMUTH_ERROR_DEG = 0.0           # СКО ошибки выставки азимута (нормальный закон), град

LATITUDE_DEG = 0.0                # широта точки входа
LONGITUDE_DEG = 0.0               # долгота точки входа
WIND_ON = True                    # False — расчёт без ветра для сравнения

WIND_MODEL = VenusWindModel(filter_order=2, latitude_deg=LATITUDE_DEG, use_weibull=True)


# ---------------------------------------------------------------------------
# Аэродинамика (ваша таблица; исправлена опечатка 0.1 -> 1.0 в сетке по Маху:
# сетка была немонотонной, и при 0.6 < M < 1.2 интерполяция шла не по тем узлам)
# ---------------------------------------------------------------------------
CX_M = [0, 0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
CX_Y = [0.75, 0.8, 0.9, 1.1, 1.3, 1.45, 1.52, 1.55, 1.6, 1.7, 1.8, 1.78, 1.75, 1.7, 1.65, 1.6, 1.55] + [1.52] * 66


def cx_func(Mach, h):
    return newton_interpolation(CX_M, CX_Y, Mach)


# ---------------------------------------------------------------------------
def entry_azimuth(wind, rng):
    """Азимут входа ψ0, рад."""
    if AZIMUTH_MODE == "fixed":
        psi = AZIMUTH_FIXED_DEG * cToRad
    elif AZIMUTH_MODE == "downwind":
        psi = downwind_azimuth(*WIND_MODEL.mean_vector(*H_ALIGN))
    elif AZIMUTH_MODE == "downwind_realization":
        psi = downwind_azimuth(*wind.mean_vector(*H_ALIGN))
    else:
        raise ValueError(f"Неизвестный AZIMUTH_MODE: {AZIMUTH_MODE}")
    return psi + rng.normal(0.0, AZIMUTH_ERROR_DEG * cToRad)


def compute_trajectory(task):
    i, seed = task
    rng = np.random.default_rng(seed)
    # порядок розыгрыша фиксирован: при одном seed ветер одинаков во всех режимах азимута
    V0 = rng.uniform(*V0_RANGE)
    tetta0 = rng.uniform(*TETTA0_RANGE_DEG) * cToRad
    wind_real = WIND_MODEL.sample(rng)
    azimuth = entry_azimuth(wind_real, rng)
    wind = wind_real if WIND_ON else None

    p = DescentParams(mass=mass, S=S, V0=V0, tetta0=tetta0, azimuth=azimuth,
                      lat0=LATITUDE_DEG * cToRad, lon0=LONGITUDE_DEG * cToRad,
                      dt=DT, save_every=SAVE_EVERY)
    r = simulate(p, cx_func, wind, tag=i)
    ld = r["landing"]
    print(f"[{i:3d}] V0 = {V0:.1f}, tetta0 = {tetta0 * cToDeg:.2f}, psi0 = {azimuth * cToDeg:.2f} -> "
          f"V = {ld['V']:.2f} м/с, L = {ld['L'] / 1000:.2f} км, Z = {ld['Z'] / 1000:.3f} км, "
          f"t = {ld['t']:.1f} с, φ = {ld['lat'] * cToDeg:.4f}°, λ = {ld['lon'] * cToDeg:.4f}°", flush=True)
    return r


if __name__ == "__main__":
    start_time = time.time()
    show = "--no-show" not in sys.argv
    seeds = np.random.SeedSequence(BASE_SEED).spawn(N_RUNS)
    tasks = list(enumerate(seeds))

    with multiprocessing.Pool(processes=min(N_RUNS, multiprocessing.cpu_count())) as pool:
        results = pool.map(compute_trajectory, tasks)

    plot_standard_set(results, show=show, prefix=None if show else "mc")
    plot_landing(results, show=show, savepath=None if show else "mc_landing.png")

    st = landing_stats(results)
    print("\nИтоги по", N_RUNS, "реализациям:")
    for key, unit, scale in (("L", "км", 1e-3), ("Z", "км", 1e-3), ("V", "м/с", 1), ("t", "с", 1),
                             ("nx_max", "g", 1), ("q_max", "Па", 1)):
        a = st[key] * scale
        print(f"  {key:7s}: M = {a.mean():10.3f}  σ = {a.std(ddof=1):9.3f}  min = {a.min():10.3f}  max = {a.max():10.3f} {unit}")

    print(f"\nВремя расчёта: {time.time() - start_time:.1f} с")
