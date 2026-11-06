"""
Модификация корпуса пенетратора: перебор полуугла конуса Qk + стохастический ветер.

Что изменилось относительно исходной версии:
  * ветер — реализация случайной функции высоты (venus_wind.py), действует через
    скорость относительно воздуха;
  * пространственная (3D) модель движения: на выходе L, H и боковое отклонение Z;
  * азимут входа выставляется по ветру в верхних слоях (AZIMUTH_MODE);
  * МЕТОД ОБЩИХ СЛУЧАЙНЫХ ЧИСЕЛ: все варианты геометрии летят в ОДНОЙ и той же
    реализации ветра и с одними НУ (COMMON_RANDOM = True). Иначе разница между
    вариантами Qk смешивается со случайным разбросом и сравнение геометрий некорректно.
    Чтобы получить статистику по каждому варианту, увеличьте N_WIND_SEEDS;
  * r2 = r1 + L·tg(Qk) (было r1 + L·Qk — для 55° это занижение радиуса в 1,5 раза);
  * Px = m / (Cx·S) (было m / Cx * S из-за приоритета операций).

Нужны файлы venus_wind.py и descent_core.py в той же папке.
"""

import math as m
import multiprocessing
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from venus_wind import VenusWindModel, downwind_azimuth
from descent_core import (DescentParams, simulate, cToRad, cToDeg, FONT,
                          plot_standard_set, landing_stats)

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
BASE_SEED = 2000
COMMON_RANDOM = True        # одна реализация ветра и НУ для всех вариантов Qk
N_WIND_SEEDS = 1            # >1 — каждый вариант считается в N реализациях ветра (общих для всех Qk)
DT = 0.01
SAVE_EVERY = 10

mass = 120.0
Long_penetrator = 0.73
r1 = 0.01
xd = 0.06
gamma_ = 0.0000             # для баллистического сделать 0

V0_RANGE = (10_900.0, 11_100.0)
TETTA0_RANGE_DEG = (-25.0, -15.0)

# Азимут входа: "downwind" — по климатическому ветру в слое H_ALIGN,
# "downwind_realization" — по ветру данной реализации, "fixed" — AZIMUTH_FIXED_DEG
AZIMUTH_MODE = "downwind"
AZIMUTH_FIXED_DEG = 90.0
H_ALIGN = (60_000.0, 100_000.0)
AZIMUTH_ERROR_DEG = 0.0

LATITUDE_DEG = 0.0
LONGITUDE_DEG = 0.0
WIND_ON = True

Qk_list = [(10.0 + 0.25 * j) * cToRad for j in range(181)]   # 10° … 55°

WIND_MODEL = VenusWindModel(filter_order=2, latitude_deg=LATITUDE_DEG, use_weibull=True)


# ---------------------------------------------------------------------------
# Ваша модель Cx(M, Qk) — без изменений, на вход сразу число Маха
# ---------------------------------------------------------------------------
def Cx_cone(r2, M, Qk):
    if M < 0.8:
        Cx_form = 0.8 * m.sin(Qk) ** 2
        Re_approx = 1e6
        Cx_friction = 0.074 / Re_approx ** 0.2 * (Long_penetrator / (2 * r2))
        return min((Cx_form + Cx_friction) * (0.7 + 0.3 * (M / 0.8)), 1.0)
    elif M < 1.2:
        Cx_subsonic = 0.8 * m.sin(Qk) ** 2
        Cx_supersonic = 2.0 * m.sin(Qk) ** 2 / m.sqrt(M ** 2 - 0.5)
        t = (M - 0.8) / 0.4
        peak_factor = 1.0 + 2.0 * m.sin(m.pi * (M - 0.9) / 0.2) ** 2
        return min(Cx_subsonic + (Cx_supersonic - Cx_subsonic) * t * peak_factor, 2.5)
    elif M < 5.0:
        beta = m.sqrt(M ** 2 - 1)
        Cx_pressure = 2.0 * m.sin(Qk) ** 2 / beta * m.sin(Qk)
        Cx_friction = 0.03 / (M * 1e6) ** 0.2 * (Long_penetrator / (2 * r2))
        return min(Cx_pressure + Cx_friction, 1.5)
    else:
        Cx_newton = 2.0 * m.sin(Qk) ** 2
        compression_factor = 1.0 + 0.5 / (1.4 * M ** 2 * m.sin(Qk) ** 2)
        return min(Cx_newton * compression_factor, 2.0)


def make_cx_func(r2, Qk):
    def cx_func(Mach, h):
        Cxa = Cx_cone(r2, Mach, Qk)
        Cya = 0.0                                   # в исходнике множитель sin(0) = 0
        alfa = (gamma_ / xd) * (Cxa / (Cya + Cxa))
        return Cxa * m.cos(alfa) + Cya * m.sin(alfa)
    return cx_func


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
    i, j_seed, seed = task
    Qk = Qk_list[i]
    r2 = r1 + Long_penetrator * m.tan(Qk)
    S = m.pi * r2 ** 2

    rng = np.random.default_rng(seed)       # при COMMON_RANDOM seed одинаковый у всех Qk
    V0 = rng.uniform(*V0_RANGE)
    tetta0 = rng.uniform(*TETTA0_RANGE_DEG) * cToRad
    wind_real = WIND_MODEL.sample(rng)
    azimuth = entry_azimuth(wind_real, rng)
    wind = wind_real if WIND_ON else None

    p = DescentParams(mass=mass, S=S, V0=V0, tetta0=tetta0, azimuth=azimuth,
                      lat0=LATITUDE_DEG * cToRad, lon0=LONGITUDE_DEG * cToRad,
                      dt=DT, save_every=SAVE_EVERY)
    r = simulate(p, make_cx_func(r2, Qk), wind, tag=(i, j_seed))
    r["Qk"], r["r2"] = Qk, r2
    ld = r["landing"]
    print(f"Qk = {Qk * cToDeg:6.2f}°, r2 = {r2:.3f} м, seed {j_seed}: V = {ld['V']:.2f} м/с, "
          f"L = {ld['L'] / 1000:.2f} км, Z = {ld['Z'] / 1000:.3f} км, t = {ld['t']:.1f} с, "
          f"nx_max = {r['nx'].max():.1f}", flush=True)
    return r


def plot_vs_qk(results, show=True):
    """Сводка по геометрии: среднее и разброс по реализациям ветра для каждого Qk."""
    nQ = len(Qk_list)
    qk = np.array(Qk_list) * cToDeg
    st = landing_stats(results)
    shape = (nQ, N_WIND_SEEDS)
    for key, title, ylabel, scale in (("V", "Скорость у поверхности", "V, м/с", 1),
                                      ("nx_max", "Максимальная перегрузка", "nx, g", 1),
                                      ("L", "Дальность до точки падения", "L, км", 1e-3),
                                      ("Z", "Боковой снос", "Z, км", 1e-3),
                                      ("t", "Время спуска", "t, с", 1)):
        a = st[key].reshape(shape) * scale
        plt.figure()
        plt.plot(qk, a.mean(1), "k")
        if N_WIND_SEEDS > 1:
            plt.fill_between(qk, a.min(1), a.max(1), color="0.8", label="min–max по реализациям ветра")
            plt.legend()
        plt.title(f"{title} от полуугла конуса", **FONT)
        plt.xlabel("Qk, град", **FONT)
        plt.ylabel(ylabel, **FONT)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        plt.grid(True)
        if show:
            plt.show()
        else:
            plt.savefig(f"qk_{key}.png", dpi=130)
            plt.close()


if __name__ == "__main__":
    start_time = time.time()
    show = "--no-show" not in sys.argv

    seeds = np.random.SeedSequence(BASE_SEED).spawn(N_WIND_SEEDS if COMMON_RANDOM else len(Qk_list) * N_WIND_SEEDS)
    tasks = []
    for i in range(len(Qk_list)):
        for j in range(N_WIND_SEEDS):
            seed = seeds[j] if COMMON_RANDOM else seeds[i * N_WIND_SEEDS + j]
            tasks.append((i, j, seed))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(compute_trajectory, tasks)

    plot_vs_qk(results, show=show)
    # полный набор графиков — по каждому 10-му варианту, иначе 181 кривая сливается
    plot_standard_set(results[::10 * N_WIND_SEEDS], show=show, prefix=None if show else "qk")

    print(f"\nВремя расчёта: {time.time() - start_time:.1f} с")
