"""
wind_model_tests.py — проверка стохастической модели ветра venus_wind.py
(двухмасштабный формирующий фильтр по высоте + преобразование к закону Вейбулла).

Запуск:  python wind_model_tests.py [N_реализаций]      (по умолчанию 4000)
Результат: протокол в консоли + рисунки test_stats.png, test_corr.png

  T1  обратная совместимость: two_scale=False == прежняя версия (venus_wind_old.py, если есть)
  T2  ξ(h) = √(1-β)ξ_L + √β ξ_S: M = 0, D = 1 на всех высотах
  T3  K(Δh) = (1-β)K_L + βK_S в слоях с постоянными параметрами
  T4  U(h): M[U] = Ū, σ[U] = σ_Вейбулл, KS-тест на всех высотах
  T5  искажение корреляции при ξ -> U (эффект Натафа)
  T6  v, w: M[v] = v̄(h, φ), σ = σ_v, σ_w; независимость u и v
  T7  интерполяция wind(h), границы сетки
  T8  азимут «по ветру», климатический вектор
  T9  широтные множители (в т.ч. полюс — нет NaN)
  T10 сдвиг ветра dU/dh: одномасштабная vs двухмасштабная модель
  T11 разброс сноса: одномасштабная vs двухмасштабная модель
"""

import math
import os
import sys
import time

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import venus_wind as vw
from descent_core import Get_ro

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
results = []


def check(name, cond, info=""):
    results.append((name, bool(cond)))
    print(f"[{'OK  ' if cond else 'FAIL'}] {name}  {info}")


def draw(model, n, seed):
    rng = np.random.default_rng(seed)
    R = [model.sample(rng) for _ in range(n)]
    return (np.array([-r.u for r in R]), np.array([r.v for r in R]), np.array([r.w for r in R]))


model = vw.VenusWindModel()
legacy = vw.VenusWindModel(two_scale=False)
h, hk, dh = model.h, model.h / 1000.0, model.dh
print(f"Сетка: {h.size} узлов, dh = {dh} м; N = {N} реализаций\n")

# T1 ------------------------------------------------------------------------
if os.path.exists("venus_wind_old.py"):
    import venus_wind_old as old
    a = old.VenusWindModel().sample(np.random.default_rng(7))
    b = legacy.sample(np.random.default_rng(7))
    d = max(np.abs(a.u - b.u).max(), np.abs(a.v - b.v).max(), np.abs(a.w - b.w).max())
    check("T1  two_scale=False совпадает с прежней версией при том же seed", d < 1e-6, f"max|Δ| = {d:.1e} м/с")
else:
    print("[--  ] T1  venus_wind_old.py не найден — пропуск")

t0 = time.time()
U, V, W = draw(model, N, 12345)
print(f"      {N} реализаций за {time.time() - t0:.1f} с")
xi_u = stats.norm.ppf(np.clip(stats.weibull_min.cdf(U, model.k, scale=model.lam), 1e-12, 1 - 1e-12))

# T2 ------------------------------------------------------------------------
se_m, se_v = 1 / math.sqrt(N), math.sqrt(2 / N)
check("T2  M[ξ] = 0 на всех высотах", np.abs(xi_u.mean(0)).max() < 5 * se_m,
      f"max|M| = {np.abs(xi_u.mean(0)).max():.3f}")
check("T2  D[ξ] = 1 на всех высотах", np.abs(xi_u.var(0) - 1).max() < 5 * se_v,
      f"max|D-1| = {np.abs(xi_u.var(0) - 1).max():.3f}")

# T3 ------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
ok3, n3 = True, 0
for a_, h0, span in zip(ax, (26.0, 50.5, 66.0, 100.0), (14.0, 4.5, 9.0, 25.0)):
    j0 = int(round(h0 * 1000 / dh))
    lags = np.arange(0, int(span * 1000 / dh))
    seg = slice(j0, j0 + lags[-1] + 1)
    const = (np.ptp(model.L_small[seg]) < 1e-6 and np.ptp(model.L_large[seg]) < 1e-6
             and np.ptp(model.beta_u[seg]) < 0.02)
    emp = np.array([np.mean(xi_u[:, j0] * xi_u[:, j0 + l]) for l in lags])
    th = model.corr_theory(j0, lags * dh)
    dev = np.abs(emp - th).max()
    if const:
        n3 += 1
        ok3 &= dev < 0.06
    a_.plot(lags * dh / 1000, emp, "o", ms=3, label="оценка")
    a_.plot(lags * dh / 1000, th, "r", label="(1-β)K_L + βK_S")
    a_.plot(lags * dh / 1000, vw.matern32_corr(lags * dh, model.L_small[j0]), "r:", label="K_S (старая модель)")
    a_.set_title(f"h₀ = {h0:.0f} км, β = {model.beta_u[j0]:.3f}" + ("" if const else " (параметры меняются)"))
    a_.set_xlabel("Δh, км")
    a_.grid(True)
    print(f"      K(Δh) h0 = {h0:5.1f} км: max|K_эмп - K_теор| = {dev:.3f}{'' if const else '  (условно)'}")
ax[0].legend(fontsize=8)
ax[0].set_ylabel("нормированная корреляция ξ_u")
fig.tight_layout()
fig.savefig("test_corr.png", dpi=120)
plt.close(fig)
check("T3  корреляционная функция = (1-β)K_L + βK_S", ok3 and n3 >= 2, f"слоёв с постоянными параметрами: {n3}")

# T4 ------------------------------------------------------------------------
mU, sU = U.mean(0), U.std(0)
rel_m = np.abs(mU - model.U_mean) / model.U_std
rel_s = np.abs(sU / model.U_std - 1)
# в двухмасштабной модели эффективное число независимых реализаций по высоте меньше —
# допуск по σ взят с запасом
check("T4  M[U] = Ū(h)", rel_m.max() < 6 / math.sqrt(N), f"max|M-Ū|/σ = {rel_m.max():.3f}")
check("T4  σ[U] = σ_Вейбулл(h)", rel_s.max() < 0.08, f"max|σ/σ_т - 1| = {rel_s.max():.3f}")
idx = np.arange(0, h.size, 40)
pv = np.array([stats.kstest(U[:, j], "weibull_min", args=(model.k[j], 0, model.lam[j])).pvalue for j in idx])
check("T4  KS-тест Вейбулла (каждые 2 км)", np.mean(pv < 0.05) < 0.15,
      f"доля p<0.05: {np.mean(pv < 0.05):.2f} (ожидается ≈0.05), min p = {pv.min():.3f}")

# T5 ------------------------------------------------------------------------
dmax = 0.0
for h0 in (1.0, 10.0, 30.0, 68.0):
    j = int(round(h0 * 1000 / dh))
    l = int(round(model.L_small[j] / dh))
    r_xi = np.corrcoef(xi_u[:, j], xi_u[:, j + l])[0, 1]
    r_u = np.corrcoef(U[:, j], U[:, j + l])[0, 1]
    dmax = max(dmax, abs(r_xi - r_u))
check("T5  искажение корреляции ξ -> U мало", dmax < 0.03, f"max|ρ_ξ - ρ_U| = {dmax:.3f}")

# T6 ------------------------------------------------------------------------
sv_err = np.abs(V.std(0) / model.sig_v - 1).max()
sw_err = np.abs(W.std(0) / model.sig_w - 1).max()
check("T6  σ[v] = σ_v(h), σ[w] = σ_w(h)", max(sv_err, sw_err) < 0.08, f"v {sv_err:.3f}, w {sw_err:.3f}")
check("T6  M[v] = 0 на экваторе", (np.abs(V.mean(0)) / model.sig_v).max() < 6 / math.sqrt(N))
cuv = np.abs([np.corrcoef(U[:, j], V[:, j])[0, 1] for j in idx]).max()
check("T6  u и v независимы", cuv < 6 / math.sqrt(N), f"max|ρ_uv| = {cuv:.3f}")
m45 = vw.VenusWindModel(latitude_deg=45.0)
_, V45, _ = draw(m45, 500, 1)
j68 = int(68_000 / dh)
check("T6  φ = 45°: M[v] на 68 км ≈ +8 м/с (к северному полюсу)",
      abs(V45[:, j68].mean() - m45.V_mean[j68]) < 4 * m45.sig_v[j68] / math.sqrt(500),
      f"M[v] = {V45[:, j68].mean():.2f}, задано {m45.V_mean[j68]:.2f} м/с")

# T7 ------------------------------------------------------------------------
r0 = model.sample(np.random.default_rng(3))
hq = np.random.default_rng(4).uniform(-1000, 140_000, 5000)
got = np.array([r0(x) for x in hq])
exp_ = np.stack([np.interp(hq, r0.h, r0.u), np.interp(hq, r0.h, r0.v), np.interp(hq, r0.h, r0.w)], 1)
check("T7  wind(h) == np.interp, вне сетки — крайние значения", np.abs(got - exp_).max() < 1e-9)

# T8 ------------------------------------------------------------------------
psi = math.degrees(vw.downwind_azimuth(*model.mean_vector(60_000, 100_000)))
check("T8  азимут по климатическому ветру на экваторе = 270°", abs(psi - 270) < 1e-9, f"ψ0 = {psi:.2f}°")
psi45 = math.degrees(vw.downwind_azimuth(*m45.mean_vector(60_000, 100_000)))
print(f"      на широте 45° азимут «по ветру» = {psi45:.2f}° (учтён снос к полюсу)")

# T9 ------------------------------------------------------------------------
r90 = vw.VenusWindModel(latitude_deg=90.0).sample(np.random.default_rng(1))
check("T9  широта 70°: Ū уменьшается вдвое", np.allclose(vw.VenusWindModel(latitude_deg=70.0).U_mean, model.U_mean * 0.5))
check("T9  широта 90°: нет NaN, U ≡ 0, v̄ = 0", np.all(np.isfinite(r90.u)) and np.abs(r90.u).max() == 0
      and np.all(np.isfinite(r90.v)))

# T10, T11 ------------------------------------------------------------------
NL = min(N, 1000)
U_old, _, _ = draw(legacy, NL, 777)
U_new = U[:NL]
shear = {"одномасштабная": np.diff(U_old, axis=1) / (dh / 1000),
         "двухмасштабная": np.diff(U_new, axis=1) / (dh / 1000)}
s_mean = np.abs(np.gradient(model.U_mean, hk))
print("\n      |dU/dh|, м/с на км (медиана / 99-й процентиль)")
print("      слой, км    сдвиг среднего   одномасштабная     двухмасштабная")
for lo, hi in ((0, 10), (15, 45), (50, 56), (60, 75)):
    msk = (hk[1:] >= lo) & (hk[1:] < hi)
    row = [f"{np.percentile(np.abs(s[:, msk]), 50):5.1f} / {np.percentile(np.abs(s[:, msk]), 99):5.1f}"
           for s in shear.values()]
    print(f"      {lo:3d}–{hi:3d}         {s_mean[1:][msk].mean():5.1f}        {row[0]}      {row[1]}")

ro = np.array([Get_ro(x) for x in h])
Vy = np.sqrt(2 * 120 * 8.87 / (ro * 0.76))     # установившаяся скорость снижения аппарата
dtdh = dh / Vy
dr_old = (U_old * dtdh).sum(1) / 1000
dr_new = (U_new * dtdh).sum(1) / 1000
print(f"\n      снос пассивной частицы по ветру: одномасштабная  M = {dr_old.mean():.0f} км, σ = {dr_old.std():.1f} км")
print(f"                                       двухмасштабная  M = {dr_new.mean():.0f} км, σ = {dr_new.std():.1f} км")
check("T11 среднее сноса не изменилось (меняется только структура разброса)",
      abs(dr_old.mean() - dr_new.mean()) < 4 * dr_new.std() / math.sqrt(NL))

# Рисунок ------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(16, 6), sharey=True)
ax[0].plot(model.U_mean, hk, "k", lw=2, label="Ū задано")
ax[0].plot(mU, hk, "r--", label="M[U] модель")
ax[0].plot(model.U_std, hk, "b", lw=2, label="σ теор.")
ax[0].plot(sU, hk, "c--", label="σ модель")
ax[0].set_xlabel("м/с")
ax[0].legend(fontsize=8)
ax[1].plot(xi_u.var(0), hk, label="D[ξ_u]")
ax[1].axvline(1, color="k", lw=0.8)
ax[1].set_xlim(0.8, 1.2)
ax[1].set_xlabel("дисперсия")
ax[1].legend(fontsize=8)
ax[2].plot(pv, hk[idx], "o", ms=3)
ax[2].axvline(0.05, color="r")
ax[2].set_xlabel("KS p-value (Вейбулл)")
for name, s, c in (("одномасштабная", shear["одномасштабная"], "0.6"), ("двухмасштабная", shear["двухмасштабная"], "C0")):
    ax[3].plot(np.percentile(np.abs(s), 99, axis=0), hk[1:], color=c, label=f"p99 |dU/dh|, {name}")
ax[3].plot(s_mean, hk, "k", label="|dŪ/dh| среднего")
ax[3].set_xlabel("м/с на км")
ax[3].legend(fontsize=8)
ax[0].set_ylabel("Высота, км")
for a_ in ax:
    a_.grid(True)
fig.suptitle(f"Проверка модели ветра (двухмасштабная), N = {N}")
fig.tight_layout()
fig.savefig("test_stats.png", dpi=120)
plt.close(fig)

print(f"\nИтого: {sum(ok for _, ok in results)} из {len(results)} проверок пройдено")
