"""
test_6dof.py — проверка 6-DOF модели (descent_6dof.py) и вращения планеты в 3-DOF (descent_core.py).
Запуск: python test_6dof.py        (≈ 1–2 мин)

  T1  кватернионы: C_IB ортогональна, переход q -> C -> q точный
  T2  вакуум: сохраняются энергия и момент импульса орбиты, энергия и кинетический
      момент вращения твёрдого тела (с нутацией)
  T3  вращение планеты: 3-DOF (уравнения в ENU с кориолисовым и центробежным членами)
      и 6-DOF (инерциальная система) дают одну траекторию — две независимые записи
  T4  симметричный устойчивый аппарат (α ≈ 0): 6-DOF совпадает с 3-DOF, с ветром и без
  T5  статический момент и балансировочный угол: α_бал = y_cm·C_A / (x_st·C_Nα)
  T6  затухание закрутки: ln p(t) = ln p0 + ∫ ρ V S d² C_lp / (4 I_x) dt
  T7  критерий динамической устойчивости C_Lα − m d²/(2I_t)·C_mq > 0
"""
import math as m
import sys

import numpy as np

import descent_core as dc
import descent_6dof as d6
import venus_wind as vw

CX_M = [0, 0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3.2, 3.6, 4, 4.4, 4.8, 5.2, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
CX_Y = [0.75, 0.8, 0.9, 1.1, 1.3, 1.45, 1.52, 1.55, 1.6, 1.7, 1.8, 1.78, 1.75, 1.7, 1.65, 1.6, 1.55] + [1.52] * 66


def cx(M, h):
    return dc.newton_interpolation(CX_M, CX_Y, M)


def cx0(M, h):
    return 0.0


def cx_tiny(M, h):          # «нет атмосферы» для 3-DOF (там делится на Cx при записи)
    return 1e-12


results = []


def check(name, ok, info=""):
    results.append(bool(ok))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}  {info}", flush=True)


# T1 ------------------------------------------------------------------------
rng = np.random.default_rng(1)
err_o = err_q = 0.0
for _ in range(200):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    C = np.array(d6.quat_to_dcm(*q))
    err_o = max(err_o, np.abs(C @ C.T - np.eye(3)).max())
    q2 = np.array(d6.dcm_to_quat(C))
    err_q = max(err_q, min(np.abs(q2 - q).max(), np.abs(q2 + q).max()))
check("T1  C_IB ортогональна, q -> C -> q", err_o < 1e-12 and err_q < 1e-12, f"{err_o:.1e}, {err_q:.1e}")

# T2 ------------------------------------------------------------------------
body = d6.Body(clp=0.0, cmq=0.0)
p = d6.Params6(V0=7_200.0, tetta0=0.05, azimuth=1.2, h0=400_000.0, spin0=7.85, rate0=(0.3, -0.2),
               omega_planet=0.0, dt_max=0.05)
s = d6.initial_state(p)


def invariants(s):
    r, v = np.array(s[:3]), np.array(s[3:6])
    E = 0.5 * v @ v - d6.MU / np.linalg.norm(r)
    h = np.cross(r, v)
    C = np.array(d6.quat_to_dcm(*s[6:10]))
    w = np.array(s[10:13])
    Iw = np.array([body.Ix, body.It, body.It]) * w
    return E, h, 0.5 * w @ Iw, C @ Iw


E0, h0_, T0, H0 = invariants(s)
t = 0.0
for _ in range(80_000):           # 1000 с с шагом 0,0125 (0,1 рад за шаг, как в simulate6)
    s = d6.rk4(t, s, 0.0125, body, cx0, None, 0.0)
    t += 0.0125
E1, h1, T1, H1 = invariants(s)
check("T2  орбита: энергия и момент импульса", abs(E1 / E0 - 1) < 1e-9 and np.abs(h1 - h0_).max() / np.linalg.norm(h0_) < 1e-9,
      f"ΔE/E = {abs(E1 / E0 - 1):.1e}")
check("T2  твёрдое тело: энергия вращения и кинетический момент в I",
      abs(T1 / T0 - 1) < 1e-3 and np.abs(H1 - H0).max() / np.linalg.norm(H0) < 1e-3,
      f"ΔT/T = {abs(T1 / T0 - 1):.1e}, ΔH/H = {np.abs(H1 - H0).max() / np.linalg.norm(H0):.1e}")

# T3, T4 --------------------------------------------------------------------
stable_sym = d6.Body(y_cm=0.0, cmq=-2.0)
wind = vw.VenusWindModel().sample(np.random.default_rng(2026))
S = m.pi * 0.8 ** 2 / 4
th0, psi0 = -20 * m.pi / 180, 1.5 * m.pi


def run3(Om, w, frame="inertial"):
    pp = dc.DescentParams(mass=120.0, S=S, V0=11_000.0, tetta0=th0, azimuth=psi0, dt=0.01,
                          omega_planet=Om, entry_frame=frame)
    return dc.simulate(pp, cx, w)["landing"]


def run6(Om, w, body=stable_sym, spin=0.0):
    pp = d6.Params6(V0=11_000.0, tetta0=th0, azimuth=psi0, omega_planet=Om, spin0=spin)
    return d6.simulate6(pp, body, cx, w)


# T3: баллистический полёт без сопротивления — эффект вращения только кинематический,
# обе записи уравнений должны совпасть с точностью интегрирования
def ball3(Om):
    pp = dc.DescentParams(mass=120.0, S=S, V0=11_000.0, tetta0=th0, azimuth=psi0, dt=0.001, save_every=1000,
                          omega_planet=Om, lat0=0.6)
    return dc.simulate(pp, cx_tiny, None)["landing"]


def ball6(Om):
    pp = d6.Params6(V0=11_000.0, tetta0=th0, azimuth=psi0, omega_planet=Om, spin0=0.0, lat0=0.6, dt_max=0.001)
    return d6.simulate6(pp, stable_sym, cx_tiny, None)["landing"]


b3_0, b3, b6 = ball3(0.0), ball3(dc.OMEGA_VENUS), ball6(dc.OMEGA_VENUS)
eff = (b3["L"] - b3_0["L"], b3["Z"] - b3_0["Z"])
check("T3  вращение планеты: 3-DOF (ENU) = 6-DOF (инерциальная), полёт без сопротивления, φ0 = 34°",
      abs(b6["L"] - b3["L"]) < 1.0 and abs(b6["Z"] - b3["Z"]) < 1.0,
      f"эффект вращения ΔL = {eff[0]:+.1f} м, ΔZ = {eff[1]:+.1f} м; расхождение моделей "
      f"{b6['L'] - b3['L']:+.2f} / {b6['Z'] - b3['Z']:+.2f} м")

l3_0 = run3(0.0, None)
l3 = run3(dc.OMEGA_VENUS, None)
r6 = run6(dc.OMEGA_VENUS, None)
l6 = r6["landing"]
print(f"      3-DOF без вращения:  L = {l3_0['L']:.1f} м, Z = {l3_0['Z']:.2f} м, t = {l3_0['t']:.2f} с")
print(f"      3-DOF с вращением:   L = {l3['L']:.1f} м, Z = {l3['Z']:.2f} м, t = {l3['t']:.2f} с")
print(f"      6-DOF с вращением:   L = {l6['L']:.1f} м, Z = {l6['Z']:.2f} м, t = {l6['t']:.2f} с, max α = {r6['alpha'].max():.3f}°")
print(f"      эффект вращения Венеры на полном спуске (3-DOF): ΔL = {l3['L'] - l3_0['L']:+.1f} м, ΔZ = {l3['Z'] - l3_0['Z']:+.2f} м")
check("T4  без ветра: 6-DOF (α≈0) = 3-DOF", abs(l6["L"] - l3["L"]) < 0.002 * l3["L"] and abs(l6["t"] - l3["t"]) < 0.002 * l3["t"]
      and abs(l6["V"] - l3["V"]) < 0.01 * l3["V"],
      f"ΔL = {l6['L'] - l3['L']:.0f} м, Δt = {l6['t'] - l3['t']:.2f} с, ΔV = {l6['V'] - l3['V']:.3f} м/с")
l3w = run3(dc.OMEGA_VENUS, wind)
r6w = run6(dc.OMEGA_VENUS, wind)
l6w = r6w["landing"]
check("T4  с ветром: 6-DOF (α≈0) = 3-DOF", abs(l6w["L"] - l3w["L"]) < 0.003 * l3w["L"] and abs(l6w["Z"] - l3w["Z"]) < 100,
      f"L {l3w['L'] / 1e3:.2f} / {l6w['L'] / 1e3:.2f} км, Z {l3w['Z']:.0f} / {l6w['Z']:.0f} м, max α = {r6w['alpha'].max():.2f}°")

# T5 ------------------------------------------------------------------------
b5 = d6.Body(y_cm=0.0)
pp = d6.Params6(V0=200.0, tetta0=-m.pi / 2 + 0.3, azimuth=0.0, h0=40_000.0, spin0=0.0, alpha0=0.02)
s5 = d6.initial_state(pp)
o = {}
d6.rhs6(0.0, s5, b5, cx, None, 0.0, o)
alpha = m.atan2(m.hypot(o["by"], o["bz"]), o["bx"])
Mt = b5.It * m.hypot(*d6.rhs6(0.0, s5, b5, cx, None, 0.0)[11:13])
Mt_th = o["q"] * b5.S * b5.cn_alpha_ratio * o["CA"] * b5.x_st * m.sin(alpha)
check("T5  восстанавливающий момент = q S C_Nα x_st sin α", abs(Mt / Mt_th - 1) < 1e-6, f"{Mt:.3f} / {Mt_th:.3f} Н·м")
b5 = d6.Body(cmq=-2.0)
r5 = run6(dc.OMEGA_VENUS, None, b5, 0.0)
sel = (r5["H"] < 30_000) & (r5["H"] > 5_000)
CA5 = r5["CX"][sel]
a_th = np.degrees(np.arctan(b5.y_cm / (b5.x_st * b5.cn_alpha_ratio)))   # C_A сокращается
check("T5  балансировочный угол без закрутки", abs(np.median(r5["alpha"][sel]) - a_th) < 0.05 * a_th,
      f"модель {np.median(r5['alpha'][sel]):.3f}°, формула {a_th:.3f}°")

# T6 ------------------------------------------------------------------------
b6 = d6.Body(y_cm=0.0, cmq=-2.0)
r6s = run6(dc.OMEGA_VENUS, None, b6, 7.85)
T, Va, H = r6s["T"], r6s["Va"], r6s["H"]
rho = np.array([d6.atmosphere(h)[0] for h in H])
rate = rho * Va * b6.S * b6.d_ref ** 2 * b6.clp / (4 * b6.Ix)
lnp_th = m.log(1.25) + np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(T))])
k = np.argmin(np.abs(T - 200.0))
check("T6  затухание закрутки по C_lp", abs(r6s["p_rps"][k] / m.exp(lnp_th[k]) - 1) < 0.01,
      f"t = 200 с: p = {r6s['p_rps'][k]:.4f} об/с, формула {m.exp(lnp_th[k]):.4f} об/с")

# T7 ------------------------------------------------------------------------
print("      критерий C_Lα − m d²/(2I_t)·C_mq на дозвуке (C_A ≈ 0,8):")
ok7 = True
for cmq in (-0.2, -0.5, -1.0, -1.5):
    b7 = d6.Body(y_cm=0.0, cmq=cmq)
    CA = 0.8
    crit = (b7.cn_alpha_ratio - 1) * CA - b7.mass * b7.d_ref ** 2 / (2 * b7.It) * cmq
    r7 = run6(dc.OMEGA_VENUS, None, b7, 0.0)
    a_low = r7["alpha"][(r7["H"] < 30_000)].max()
    print(f"        C_mq = {cmq:5.1f}: критерий {crit:+.3f} -> {'устойчив' if crit > 0 else 'неустойчив'};"
          f"  max α ниже 30 км = {a_low:.2f}°")
    ok7 &= (crit > 0) == (a_low < 5.0)
check("T7  устойчивость по модели совпадает с критерием", ok7)

print(f"\nИтого: {sum(results)} из {len(results)}")
sys.exit(0 if all(results) else 1)
