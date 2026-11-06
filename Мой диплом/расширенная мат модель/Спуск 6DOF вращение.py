"""
Спуск пенетратора как твёрдого тела (6 степеней свободы): вращение Венеры, стохастический
ветер, закрутка аппарата вокруг продольной оси.

Режимы (MODE):
  "single" — одна траектория: угол атаки, закрутка и резонанс, сравнение с 3-DOF;
  "mc"     — Монте-Карло для нескольких скоростей закрутки SPINS_RPS на ОДНИХ И ТЕХ ЖЕ
             реализациях (НУ, ветер, начальный крен) — метод общих случайных чисел;
  "scan"   — перебор начальной закрутки: где проходит резонанс и какой угол атаки он даёт.

Запуск:  python "Спуск 6DOF вращение.py"            (графики на экран)
         python "Спуск 6DOF вращение.py" --no-show  (графики в PNG)

Нужны файлы venus_wind.py, descent_core.py, descent_6dof.py в той же папке.
"""
import math as m
import multiprocessing
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import descent_core as dc
import descent_6dof as d6
from venus_wind import VenusWindModel, downwind_azimuth

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
MODE = "single"                 # "single", "mc", "scan"
N_RUNS = 30                     # реализаций на каждую закрутку (режим "mc")
BASE_SEED = 2026
SPIN_RPS = 2.0                 # об/с, начальная закрутка: пенетраторы «Марс-96» — 1,25 об/с
SPINS_RPS = [0.0, 1.25, 2.5]    # режим "mc": какие закрутки сравнивать
SCAN_RPS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

WIND_ON = True
ROTATION_ON = True              # вращение Венеры
LATITUDE_DEG = 0.0
LONGITUDE_DEG = 0.0
V0_RANGE = (10_900.0, 11_100.0)
TETTA0_RANGE_DEG = (-21.0, -19.0)
H_ALIGN = (60_000.0, 100_000.0)  # азимут входа — по климатическому ветру в этом слое

# Аппарат. Значения по умолчанию — в descent_6dof.Body (там же откуда они взяты);
# здесь можно переопределить любое, например BODY = d6.Body(y_cm=0.0, cmq=-0.3)
BODY = d6.Body()

# таблица Cx(M) из скрипта Монте-Карло (осевая сила при малых углах атаки)
CX_M = [0, 0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
CX_Y = [0.75, 0.8, 0.9, 1.1, 1.3, 1.45, 1.52, 1.55, 1.6, 1.7, 1.8, 1.78, 1.75, 1.7, 1.65, 1.6, 1.55] + [1.52] * 66

WIND_MODEL = VenusWindModel(latitude_deg=LATITUDE_DEG)
ALPHA_VALID_DEG = 30.0          # выше — линейная аэродинамика (C_N = C_Nα sinα) недостоверна


def cx_func(Mach, h):
    return dc.newton_interpolation(CX_M, CX_Y, Mach)


# ---------------------------------------------------------------------------
def draw(seed):
    """Случайные величины одной реализации: НУ, ветер, начальный крен."""
    rng = np.random.default_rng(seed)
    V0 = rng.uniform(*V0_RANGE)
    tetta0 = rng.uniform(*TETTA0_RANGE_DEG) * dc.cToRad
    wind = WIND_MODEL.sample(rng)
    roll0 = rng.uniform(0.0, 2 * m.pi)
    return V0, tetta0, wind, roll0


def run6(task):
    i, seed, spin_rps = task
    V0, tetta0, wind, roll0 = draw(seed)
    psi0 = downwind_azimuth(*WIND_MODEL.mean_vector(*H_ALIGN))
    p = d6.Params6(V0=V0, tetta0=tetta0, azimuth=psi0, lat0=LATITUDE_DEG * dc.cToRad,
                   lon0=LONGITUDE_DEG * dc.cToRad, spin0=spin_rps * 2 * m.pi, roll0=roll0,
                   omega_planet=dc.OMEGA_VENUS if ROTATION_ON else 0.0)
    r = d6.simulate6(p, BODY, cx_func, wind if WIND_ON else None, tag=(i, spin_rps))
    ld = r["landing"]
    hi = r["H"] > 60_000
    r["alpha_max_hi"] = float(r["alpha"][hi].max())
    r["alpha_max_lo"] = float(r["alpha"][~hi].max())
    print(f"[{i:3d}] p0 = {spin_rps:4.2f} об/с: L = {ld['L'] / 1e3:7.2f} км, Z = {ld['Z'] / 1e3:7.3f} км, "
          f"V = {ld['V']:5.2f} м/с, α у поверхности = {ld['alpha']:5.2f}°, max α выше/ниже 60 км = "
          f"{r['alpha_max_hi']:5.1f}° / {r['alpha_max_lo']:5.1f}°", flush=True)
    return r


def run3(seed):
    V0, tetta0, wind, _ = draw(seed)
    psi0 = downwind_azimuth(*WIND_MODEL.mean_vector(*H_ALIGN))
    p = dc.DescentParams(mass=BODY.mass, S=BODY.S, V0=V0, tetta0=tetta0, azimuth=psi0,
                         lat0=LATITUDE_DEG * dc.cToRad, lon0=LONGITUDE_DEG * dc.cToRad,
                         omega_planet=dc.OMEGA_VENUS if ROTATION_ON else 0.0)
    return dc.simulate(p, cx_func, wind if WIND_ON else None)


def report_body():
    b = BODY
    print(f"Аппарат: m = {b.mass} кг, d = {b.d_ref} м, Ix = {b.Ix:.2f}, It = {b.It:.2f} кг·м², "
          f"x_st = {b.x_st} м, y_cm = {b.y_cm * 1000:.2f} мм, C_Nα/C_A = {b.cn_alpha_ratio}, "
          f"C_mq = {b.cmq}, C_lp = {b.clp}")
    for CA, lab in ((1.52, "гиперзвук"), (0.8, "дозвук")):
        c = b.stability_criterion(CA)
        print(f"  критерий динамической устойчивости ({lab}, C_A = {CA}): {c:+.3f} -> "
              f"{'колебания затухают' if c > 0 else 'КОЛЕБАНИЯ РАСТУТ'}")
    if b.y_cm:
        print(f"  балансировочный угол от смещения ЦМ: {m.degrees(m.atan(b.y_cm / (b.x_st * b.cn_alpha_ratio))):.2f}°")


def save_or_show(name, show):
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig(name, dpi=130)
        plt.close()


def warn_alpha(results):
    bad = [r for r in results if max(r["alpha_max_hi"], r["alpha_max_lo"]) > ALPHA_VALID_DEG]
    if bad:
        print(f"\nВНИМАНИЕ: в {len(bad)} из {len(results)} траекторий угол атаки превышал {ALPHA_VALID_DEG:.0f}°: "
              "аэродинамика с линейной нормальной силой там недостоверна, а сам факт — признак "
              "резонанса или динамической неустойчивости.")


# ---------------------------------------------------------------------------
def mode_single(show):
    seed = np.random.SeedSequence(BASE_SEED).spawn(1)[0]
    r = run6((0, seed, SPIN_RPS))
    r3 = run3(seed)
    l3 = r3["landing"]
    print(f"  3-DOF (α = 0):  L = {l3['L'] / 1e3:7.2f} км, Z = {l3['Z'] / 1e3:7.3f} км, V = {l3['V']:5.2f} м/с, t = {l3['t']:.1f} с")
    print(f"  6-DOF:          t = {r['landing']['t']:.1f} с; max ω_res = {np.nanmax(r['omega_res']):.2f} рад/с "
          f"= {np.nanmax(r['omega_res']) / (2 * m.pi):.2f} об/с")
    warn_alpha([r])
    F = dc.FONT
    plt.figure()
    plt.plot(r["alpha"], r["H"] / 1000, "k", lw=0.8)
    plt.title("Пространственный угол атаки", **F)
    plt.xlabel("α, град", **F)
    plt.ylabel("Высота, км", **F)
    save_or_show("6dof_alpha_h.png", show)

    sel = r["T"] < 200
    plt.figure()
    plt.plot(r["T"][sel], r["p_rps"][sel] * 2 * m.pi, "k", label="закрутка p")
    plt.plot(r["T"][sel], r["omega_res"][sel], "r", label=r"$\omega_n/\sqrt{1-I_x/I_t}$")
    plt.plot(r["T"][sel], r["alpha"][sel] / 10, color="0.6", lw=0.8, label="α / 10, град")
    plt.title("Резонанс: закрутка и частота колебаний", **F)
    plt.xlabel("Время, с", **F)
    plt.ylabel("рад/с", **F)
    plt.legend()
    save_or_show("6dof_resonance.png", show)

    plt.figure()
    plt.plot(r3["L"] / 1000, r3["Z"] / 1000, "b", label="3-DOF (α = 0)")
    plt.plot(r["L"] / 1000, r["Z"] / 1000, "k", label=f"6-DOF, закрутка {SPIN_RPS} об/с")
    plt.title("Проекция траектории на поверхность", **F)
    plt.xlabel("Дальность L, км", **F)
    plt.ylabel("Боковое отклонение Z, км", **F)
    plt.legend()
    save_or_show("6dof_plan.png", show)
    dc.plot_standard_set([r], show=show, prefix=None if show else "6dof")


def ellipse(L, Z):
    C = np.cov(np.vstack([L, Z]))
    w, v = np.linalg.eigh(C)
    a = np.linspace(0, 2 * np.pi, 200)
    e = v @ (3 * np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(a), np.sin(a)]))
    return L.mean() + e[0], Z.mean() + e[1]


def mode_mc(show):
    seeds = np.random.SeedSequence(BASE_SEED).spawn(N_RUNS)
    tasks = [(i, s, sp) for sp in SPINS_RPS for i, s in enumerate(seeds)]
    with multiprocessing.Pool() as pool:
        res = pool.map(run6, tasks)
        res3 = pool.map(run3, seeds)
    groups = {sp: res[k * N_RUNS:(k + 1) * N_RUNS] for k, sp in enumerate(SPINS_RPS)}
    print("\nИтоги (одни и те же НУ, ветер и начальный крен для всех вариантов):")
    print("  вариант           M[L], км  σL, км   M[Z], км  σZ, км   M[V], м/с  max α > 60 км  max α < 60 км")
    rows = [("3-DOF, α = 0", [r["landing"] for r in res3], None)]
    rows += [(f"6-DOF, {sp:4.2f} об/с", [r["landing"] for r in g], g) for sp, g in groups.items()]
    plt.figure()
    for name, lands, g in rows:
        L = np.array([x["L"] for x in lands]) / 1000
        Z = np.array([x["Z"] for x in lands]) / 1000
        V = np.array([x["V"] for x in lands])
        ah = f"{np.mean([r['alpha_max_hi'] for r in g]):6.1f}°" if g else "     —"
        al = f"{np.mean([r['alpha_max_lo'] for r in g]):6.1f}°" if g else "     —"
        print(f"  {name:16s} {L.mean():9.2f} {L.std(ddof=1):7.2f} {Z.mean():10.3f} {Z.std(ddof=1):7.3f} "
              f"{V.mean():10.2f}   {ah} (ср.)    {al} (ср.)")
        pl = plt.plot(L, Z, "o", ms=3.5, label=name)
        plt.plot(*ellipse(L, Z), color=pl[0].get_color(), lw=1)
    warn_alpha(res)
    plt.title("Рассеивание точек падения, эллипсы 3σ", **dc.FONT)
    plt.xlabel("Дальность L, км", **dc.FONT)
    plt.ylabel("Боковое отклонение Z, км", **dc.FONT)
    plt.legend(fontsize=9)
    save_or_show("6dof_mc_landing.png", show)


def mode_scan(show):
    seed = np.random.SeedSequence(BASE_SEED).spawn(1)[0]
    with multiprocessing.Pool() as pool:
        res = pool.map(run6, [(k, seed, sp) for k, sp in enumerate(SCAN_RPS)])
    w_res = max(np.nanmax(r["omega_res"]) for r in res) / (2 * m.pi)
    print(f"\nНаибольшая резонансная частота на траектории: {w_res:.2f} об/с. Закрутка ниже неё проходит "
          f"через резонанс; выше — нет.")
    plt.figure()
    plt.plot(SCAN_RPS, [r["alpha_max_hi"] for r in res], "ko-", label="max α выше 60 км")
    plt.axvline(w_res, color="r", ls="--", label="max резонансной частоты")
    plt.title("Угол атаки от начальной закрутки", **dc.FONT)
    plt.xlabel("Начальная закрутка, об/с", **dc.FONT)
    plt.ylabel("α, град", **dc.FONT)
    plt.legend()
    save_or_show("6dof_spin_scan.png", show)


if __name__ == "__main__":
    t0 = time.time()
    show = "--no-show" not in sys.argv
    report_body()
    {"single": mode_single, "mc": mode_mc, "scan": mode_scan}[MODE](show)
    print(f"\nВремя расчёта: {time.time() - t0:.1f} с")
