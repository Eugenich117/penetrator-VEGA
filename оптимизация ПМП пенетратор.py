#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 ПМП-ОПТИМИЗАЦИЯ СПУСКА ПЕНЕТРАТОРА В АТМОСФЕРУ ВЕНЕРЫ
 Источники данных: оптимизация_пенетратор.py, модификация_корпуса_пенетратора.py
=============================================================================
...
"""

import math, sys, time                           # [CROSS-PLATFORM] добавлен sys
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.font_manager as _fm            # [CROSS-PLATFORM] для поиска шрифта
import matplotlib.pyplot as plt
import warnings
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')

# [CROSS-PLATFORM] Выбор шрифта с fallback-цепочкой
def _select_font():
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in ('Times New Roman', 'DejaVu Serif', 'Georgia', 'serif'):
        if name in available:
            return name
    return 'serif'

matplotlib.rcParams.update({'font.family': _select_font(), 'font.size': 13})

# [CROSS-PLATFORM] Контекст мультипроцессинга:
#   Windows          → spawn (единственный доступный)
#   macOS (py≥3.8)   → spawn (fork нестабилен из-за системных потоков)
#   Linux            → fork  (быстрее, безопасен)
def _mp_ctx():
    if sys.platform in ('win32', 'darwin'):
        return _mp.get_context('spawn')
    return _mp.get_context('fork')

# ─── Параметры аппарата ───────────────────────────────────────────────────────
Rb        = 6_051_800.0
g0        = 8.87
mass      = 120.0
d_body    = 0.8
S         = math.pi * (d_body / 2) ** 2
r_nose    = 0.5

h0        = 125_000.0
V0        = 11_000.0
theta0    = -19.0 * math.pi / 180.0

h_conv    = 80.0
T_wall    = 600.0
R_CO2     = 188.9

lambda_V  = 1.0
lambda_Q  = 5.0e-6

K_MAX     = 0.30

DT        = 0.2
DT_FINE   = 0.02
T_MAX     = 3_500.0
T_EST     = 130.0

DEG = math.pi / 180.0

# ─── Таблицы атмосферы ────────────────────────────────────────────────────────
_H = [130,128,126,124,122,120,118,116,114,112,110,108,106,104,102,
      100,98,96,94,92,90,88,86,84,82,80,78,76,74,72,
      70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,
      40,38,36,34,32,31,30,29,28,27,26,25,24,23,22,
      21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,
      6,5,4,3,2,1,0]

_RHO = [3.97200000e-08,7.890e-7,1.35931821e-06,1.77164551e-06,2.30904567e-06,
        3.00945751e-06,3.92232801e-05,5.11210308e-05,6.66277727e-05,8.68382352e-05,
        1.13179216e-04,1.47510310e-04,1.92255188e-04,2.50572705e-04,3.26579903e-04,
        1.347e-4,0.0002,0.0004,0.0007,0.0012,0.0019,0.0031,0.0049,0.0077,
        0.0119,0.0178,0.0266,0.0393,0.0578,0.0839,
        0.1210,0.1729,0.2443,0.3411,0.4694,0.6289,
        0.8183,1.0320,1.2840,1.5940,1.9670,2.4260,2.9850,3.6460,4.4040,
        5.2760,6.2740,7.4200,8.7040,9.4060,10.1500,10.9300,11.7700,12.6500,
        13.5900,14.5700,15.6200,16.7100,17.8800,19.1100,20.3900,21.7400,
        23.1800,24.6800,26.2700,27.9500,29.7400,31.6000,33.5400,35.5800,
        37.7200,39.9500,42.2600,44.7100,47.2400,49.8700,52.6200,55.4700,
        58.4500,61.5600,64.7900]

_VS = [174,176,178,180,182,185,186,187,190,193,195,196,198,199,201,
       203,205,206,208.0,208.0,209.0,212.2,215.4,218.6,221.8,225.0,
       228.2,231.4,234.6,237.8,241.0,244.0,247.0,250.0,253.0,256.0,
       263.2,270.4,277.6,284.8,292.0,296.8,301.6,306.4,311.2,316.0,
       321.2,326.4,331.6,336.8,339.4,342.0,344.6,347.2,349.8,352.4,
       355.0,357.4,359.8,362.2,364.6,367.0,369.4,371.8,374.2,376.6,
       379.0,381.0,383.0,385.0,387.0,389.0,391.2,393.4,395.6,397.8,
       400.0,402.0,404.0,406.0,408.0,410.0]

_PRES = [0.0019907,0.005,0.10,0.30,0.50,0.70,0.90,1.10,1.30,1.50,1.70,
         1.90,2.10,2.30,2.50,2.66,4.45,7.519,12.81,20.0,40.0,60.0,
         110.0,170.0,280.0,450.0,700.0,1080.0,1650.0,2480.0,3690.0,5450.0,
         7970.0,11560.0,16590.0,23570.0,33060.0,45590.0,61600.0,81670.0,
         106600.0,137500.0,175600.0,222600.0,280200.0,350100.0,434200.0,
         534600.0,653700.0,794000.0,872900.0,958100.0,1050000.0,1149000.0,
         1256000.0,1370000.0,1493000.0,1625000.0,1766000.0,1917000.0,
         2079000.0,2252000.0,2436000.0,2633000.0,2843000.0,3066000.0,
         3304000.0,3557000.0,3826000.0,4112000.0,4416000.0,4739000.0,
         5081000.0,5444000.0,5828000.0,6235000.0,6665000.0,7120000.0,
         7601000.0,8109000.0,8645000.0,9210000.0]

_CX_M = [0.0,0.2,0.4,0.6,1.0,1.2,1.4,1.8,2.0,2.2,2.4,2.6,2.8,
         3.2,3.6,4.0,4.4,4.8,5.2,6.0]
_CX_V = [0.75,0.9,1.1,1.3,1.45,1.52,1.55,1.6,1.7,1.8,1.78,
         1.75,1.7,1.65,1.6,1.55,1.52,1.52,1.52,1.52]

# ─── Интерполяция Ньютона ─────────────────────────────────────────────────────

def _n4(x_tbl, y_tbl, xi):
    n = len(x_tbl)
    idx = None
    for i in range(n):
        if xi - 2 <= x_tbl[i] <= xi + 2:
            if i <= 2:       idx = list(range(4))
            elif i >= n - 2: idx = list(range(n-4, n))
            else:            idx = list(range(i-2, i+2))
            break
    if idx is None:
        idx = list(range(4)) if xi > x_tbl[0] else list(range(n-4, n))
    xp = [x_tbl[k] for k in idx]
    yp = [y_tbl[k] for k in idx]
    for j in range(1, 4):
        for i in range(3, j-1, -1):
            if xp[i] != xp[i-j]:
                yp[i] = (yp[i] - yp[i-1]) / (xp[i] - xp[i-j])
    res = yp[3]
    for i in range(2, -1, -1):
        res = res * (xi - xp[i]) + yp[i]
    return res

def _n2(x_tbl, y_tbl, xi):
    for i in range(len(x_tbl)):
        if x_tbl[i] >= xi:
            i0 = max(i-1, 0); i1 = min(i, len(x_tbl)-1)
            if i == 0: i0, i1 = 0, 1
            x0, x1 = x_tbl[i0], x_tbl[i1]
            if x1 == x0: return y_tbl[i0]
            return y_tbl[i0] + (y_tbl[i1]-y_tbl[i0])/(x1-x0)*(xi-x0)
    return y_tbl[-1]

# ─── Функции атмосферы ────────────────────────────────────────────────────────

def Get_ro(h_m):
    h_km = max(0., min(130., h_m/1000.))
    return max(_n4(_H, _RHO, h_km), 1e-14)

def v_sound(h_m):
    h_km = max(0., min(130., h_m/1000.))
    return max(_n4(_H, _VS, h_km), 1.)

def pressure_atm(h_m):
    h_km = max(0., min(130., h_m/1000.))
    return max(_n4(_H, _PRES, h_km), 0.)

def T_atm(h_m):
    ro = Get_ro(h_m)
    return pressure_atm(h_m) / (ro * R_CO2)

def dro_dh(h_m):
    eps = 500.
    return (Get_ro(min(h_m+eps,130000.)) - Get_ro(max(h_m-eps,0.))) / (2.*eps)

def Cx_func(V, h_m):
    M = V / v_sound(h_m)
    return max(0.5, min(_n2(_CX_M, _CX_V, M), 2.5))

# ─── Тепловой поток ───────────────────────────────────────────────────────────

def heat_flux(h_m, V):
    ro = Get_ro(h_m)
    q_turb = (1.15e6) * (ro**0.8 / (0.5**0.2)) * (V/7328.)**3.19
    q_comp = (7.845 * 0.5) * (ro/64.79) * (V/1000.)**8
    q_amb  = h_conv * max(0., T_atm(h_m) - T_wall)
    return max(0., q_turb + q_comp + q_amb)

def dq_dV(h_m, V):
    ro = Get_ro(h_m)
    dt = (1.15e6)*(ro**0.8/(0.5**0.2))*(3.19/7328.)*(V/7328.)**2.19
    dc = (7.845*0.5)*(ro/64.79)*(8./1000.)*(V/1000.)**7.
    return dt + dc

def dq_dro(h_m, V):
    ro = Get_ro(h_m)
    dt = (1.15e6)*(0.8*ro**(-0.2)/(0.5**0.2))*(V/7328.)**3.19
    dc = (7.845*0.5)/64.79*(V/1000.)**8
    T_a = T_atm(h_m)
    da = h_conv * (-pressure_atm(h_m)/(ro**2*R_CO2)) if T_a > T_wall else 0.
    return dt + dc + da

# ─── Уравнения движения ───────────────────────────────────────────────────────

def f_eom(V, th, R, K):
    h   = R - Rb
    ro  = Get_ro(h)
    cx  = Cx_func(V, h)
    gR  = g0*(Rb/R)**2
    Dm  = 0.5*ro*V**2*cx*S/mass
    fV  = -gR*math.sin(th) - Dm
    fth = -(gR/V)*math.cos(th) + V/R + K*Dm/V
    fR  = V*math.sin(th)
    fQ  = heat_flux(h, V)
    return fV, fth, fR, fQ

def Dm_V(V, R):
    h = R-Rb; ro = Get_ro(h); cx = Cx_func(V,h)
    return 0.5*ro*V*cx*S/mass

def sigma(V, R, pt):
    return pt * Dm_V(V, R)

def bang_bang(sg):
    return K_MAX if sg < 0. else 0.

# ─── Сопряжённые уравнения ────────────────────────────────────────────────────

def adjoint_rhs(V, th, R, pV, pt, pR, K):
    h    = R-Rb
    ro   = Get_ro(h); cx = Cx_func(V,h)
    gR   = g0*(Rb/R)**2
    Dm   = 0.5*ro*V**2*cx*S/mass
    dr   = dro_dh(h)
    dDmR = Dm*dr/max(ro,1e-14)

    dHdV  = ( pV*(-2.*Dm/V)
            + pt*(gR/V**2*math.cos(th) + 1./R + K*Dm/V**2)
            + pR*math.sin(th)
            + lambda_Q*dq_dV(h,V) )

    dHdth = ( pV*(-gR*math.cos(th))
            + pt*(gR/V*math.sin(th))
            + pR*V*math.cos(th) )

    dHdR  = ( pV*(2.*gR/R*math.sin(th) - dDmR)
            + pt*(2.*gR/(V*R)*math.cos(th) - V/R**2 + K*dDmR/V)
            + lambda_Q*dq_dro(h,V)*dr )

    return -dHdV, -dHdth, -dHdR

def hamiltonian(V, th, R, pV, pt, pR, K):
    fV,fth,fR,fQ = f_eom(V,th,R,K)
    return pV*fV + pt*fth + pR*fR + lambda_Q*fQ

# ─── Интегрирование траектории ────────────────────────────────────────────────

def integrate_traj(K_sc, t_sw, save=False):
    V=V0; th=theta0; R=Rb+h0; Q=0.; t=0.
    sw = sorted(t_sw)
    def curK(t_):
        idx = sum(1 for ts in sw if t_>=ts)
        return K_sc[min(idx, len(K_sc)-1)]

    if save:
        hist = {k:[] for k in ('t','V','theta_deg','h','Q','K')}
        def rec():
            hist['t'].append(t); hist['V'].append(V)
            hist['theta_deg'].append(th/DEG); hist['h'].append(R-Rb)
            hist['Q'].append(Q); hist['K'].append(curK(t))
        rec()

    while R > Rb and t < T_MAX:
        K = curK(t)
        h_= R-Rb
        dt = DT_FINE if h_ < 3000 else DT
        for ts in sw:
            if t < ts < t+dt: dt = ts-t+1e-9

        k1 = f_eom(V,th,R,K)
        k2 = f_eom(V+.5*dt*k1[0], th+.5*dt*k1[1], R+.5*dt*k1[2], K)
        k3 = f_eom(V+.5*dt*k2[0], th+.5*dt*k2[1], R+.5*dt*k2[2], K)
        k4 = f_eom(V+   dt*k3[0], th+   dt*k3[1], R+   dt*k3[2], K)
        c = dt/6.
        V  += c*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        th += c*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        R  += c*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        Q  += c*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
        t  += dt
        if R<=Rb: R=Rb
        if save: rec()

    if save:
        for k in hist: hist[k]=np.array(hist[k])
        return V,th,R,Q,t,hist
    return V,th,R,Q,t

# ─── Обратное интегрирование сопряжённых ─────────────────────────────────────

def integrate_adjoint_bwd(hist):
    ta = hist['t']; T = float(ta[-1])
    Vi  = interp1d(ta, hist['V'],            kind='linear', fill_value='extrapolate')
    thi = interp1d(ta, hist['theta_deg']*DEG,kind='linear', fill_value='extrapolate')
    Ri  = interp1d(ta, hist['h']+Rb,         kind='linear', fill_value='extrapolate')
    Ki  = interp1d(ta, hist['K'],            kind='previous',fill_value='extrapolate')

    def rhs(tau, ps):
        tf = T-tau
        V_=float(Vi(tf)); th_=float(thi(tf)); R_=float(Ri(tf)); K_=float(Ki(tf))
        dV,dt,dR = adjoint_rhs(V_,th_,R_,ps[0],ps[1],ps[2],K_)
        return [-dV,-dt,-dR]

    sol = solve_ivp(rhs, [0.,T], [-lambda_V,0.,0.],
                    method='Radau', dense_output=True,
                    rtol=1e-6, atol=1e-9, max_step=2.)

    ord_ = np.argsort(T-sol.t)
    t_  = (T-sol.t)[ord_]
    pV_ = sol.y[0][ord_]; pt_ = sol.y[1][ord_]; pR_ = sol.y[2][ord_]
    Sg_ = np.array([pt_[i]*Dm_V(float(Vi(ti)),float(Ri(ti)))
                    for i,ti in enumerate(t_)])
    return {'t':t_,'pV':pV_,'pt':pt_,'pR':pR_,'Sigma':Sg_}

# ─── Оптимизация ─────────────────────────────────────────────────────────────

def criterion(V,Q): return lambda_Q*Q - lambda_V*V

def cost(p):
    tau1,tau2,flg = p
    K0 = K_MAX if flg>=0.5 else 0.; K1=K_MAX-K0
    t1,t2 = tau1*T_EST, tau2*T_EST
    if tau1<tau2: Ksc=[K0,K1,K0]; tsw=[t1,t2]
    else:         Ksc=[K0,K1];    tsw=[t1]
    try:
        V_,_,_,Q_,_ = integrate_traj(Ksc,tsw,save=False)
    except: return 1e10
    if not(0<V_<20000) or Q_<0: return 1e10
    return criterion(V_,Q_)

def solve_pmp():
    sep = '═'*64
    print(sep)
    print('  ПМП-ОПТИМИЗАЦИЯ СПУСКА ПЕНЕТРАТОРА (ВЕНЕРА)')
    print(f'  V₀={V0:.0f} м/с   θ₀={theta0/DEG:.0f}°   h₀={h0/1000:.0f} км')
    print(f'  λ_V={lambda_V}   λ_Q={lambda_Q:.1e}   K_max={K_MAX}')
    print(f'  Нагрев: q_turb + q_comp + q_amb(атм. Венеры)')
    print(sep)

    Vb,_,_,Qb,Tb = integrate_traj([0.],[],save=False)
    print(f'  Баллист.:  T={Tb:.0f}с  V={Vb:.2f}м/с  Q={Qb/1e6:.4f}МДж/м²  '
          f'J={criterion(Vb,Qb):+.4e}')

    print('\nФаза 1 — дифференциальная эволюция ...')
    t0=time.time()

    # [CROSS-PLATFORM] Используем кросс-платформенный контекст
    _ctx  = _mp_ctx()
    _pool = _ctx.Pool()
    rd = differential_evolution(cost, [(0.,1.)]*3,
         maxiter=800, tol=1e-12, seed=17, popsize=20,
         mutation=(0.4,1.6), recombination=0.9,
         updating='deferred',
         workers=_pool.map,           # Pool.map пикабелен на любой платформе
         disp=False, polish=False)
    _pool.close()
    _pool.join()
    print(f'  {time.time()-t0:.1f}с  J={rd.fun:.6e}  p={np.round(rd.x,4)}')

    print('Фаза 2 — Нелдер-Мид ...')
    t0=time.time()
    rn = minimize(cost, rd.x, method='Nelder-Mead',
                  options=dict(xatol=1e-13,fatol=1e-15,maxiter=200_000,adaptive=True))
    print(f'  {time.time()-t0:.1f}с  J={rn.fun:.6e}  p={np.round(rn.x,4)}')

    tau1,tau2,flg = rn.x
    K0=K_MAX if flg>=0.5 else 0.; K1=K_MAX-K0
    t1,t2=tau1*T_EST,tau2*T_EST
    if tau1<tau2:
        Ksc=[K0,K1,K0]; tsw=sorted([t1,t2])
        print(f'\n  K={K0:.2f} →[{tsw[0]:.2f}с]→ K={K1:.2f} →[{tsw[1]:.2f}с]→ K={K0:.2f}')
    else:
        Ksc=[K0,K1]; tsw=[t1]
        print(f'\n  K={K0:.2f} →[{tsw[0]:.2f}с]→ K={K1:.2f}')
    return Ksc,tsw,rn

# ─── Графики ──────────────────────────────────────────────────────────────────

def plot_results(ho, adj, hb, Ksc, tsw):
    VT=ho['V'][-1]; QT=ho['Q'][-1]
    Vb=hb['V'][-1]; Qb=hb['Q'][-1]
    Tf=ho['t'][-1]
    iT = np.argmin(np.abs(adj['t']-Tf))
    pVT,ptT,pRT = adj['pV'][iT],adj['pt'][iT],adj['pR'][iT]

    sep='─'*64
    print(f'\n{sep}')
    print(f'  РЕЗУЛЬТАТЫ  (h_финал={ho["h"][-1]/1000:.2f}км)')
    print(sep)
    print(f'  {"":38s} ПМП-опт.   Баллист.')
    print(f'  Скорость V(T), м/с        {VT:>12.3f} {Vb:>10.3f}')
    print(f'  Нагрев Q(T), МДж/м²       {QT/1e6:>12.4f} {Qb/1e6:>10.4f}')
    print(f'  Критерий J                {criterion(VT,QT):>+12.4e} {criterion(Vb,Qb):>+10.4e}')
    print(f'  ΔV  (ПМП−баллист.), м/с   {VT-Vb:>+12.3f}')
    print(f'  ΔQ  (ПМП−баллист.), МДж/м² {(QT-Qb)/1e6:>+11.4f}')
    HT=hamiltonian(VT,ho['theta_deg'][-1]*DEG,Rb+ho['h'][-1],pVT,ptT,pRT,ho['K'][-1])
    print(f'\n  Верификация ПМП §11.3:')
    print(f'  ψ_V(T)={pVT:+.5f}  (цель −λ_V={-lambda_V:.2f})')
    print(f'  ψ_θ(T)={ptT:+.5f}  (цель 0)')
    print(f'  ψ_R(T)={pRT:+.4e} (цель 0)')
    print(f'  H(T)  ={HT:+.4e}  (→0 при свободном T)')
    print(sep)

    sw_s=', '.join(f't={s:.1f}с' for s in tsw)
    sup=(f'ПМП-оптимальный вход пенетратора в атмосферу Венеры  '
         f'[λ_V={lambda_V}, λ_Q={lambda_Q:.1e}, K_max={K_MAX}]\n'
         f'Переключения: {sw_s}  |  V(T)={VT:.1f}м/с  Q(T)={QT/1e6:.4f}МДж/м²')

    to=ho['t']; tb=hb['t']

    fig1,axs=plt.subplots(2,3,figsize=(17,10)); fig1.suptitle(sup,fontsize=10)

    ax=axs[0,0]
    ax.plot(to,ho['h']/1000,'b-',lw=2,label='ПМП-опт.')
    ax.plot(tb,hb['h']/1000,'k--',lw=1.5,alpha=.7,label='Баллист.')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Высота, км')
    ax.set_title('Высота от времени'); ax.legend(fontsize=10); ax.grid(True)

    ax=axs[0,1]
    ax.plot(ho['h']/1000,ho['V']/1000,'b-',lw=2,label='ПМП-опт.')
    ax.plot(hb['h']/1000,hb['V']/1000,'k--',lw=1.5,alpha=.7,label='Баллист.')
    ax.set_xlabel('Высота, км'); ax.set_ylabel('Скорость, км/с')
    ax.set_title('Скорость от высоты'); ax.legend(fontsize=10); ax.grid(True)

    ax=axs[0,2]
    ax.plot(to,ho['theta_deg'],'g-',lw=2,label='ПМП-опт.')
    ax.plot(tb,hb['theta_deg'],'k--',lw=1.5,alpha=.7,label='Баллист.')
    ax.set_xlabel('Время, с'); ax.set_ylabel('θ, °')
    ax.set_title('Траекторный угол'); ax.legend(fontsize=10); ax.grid(True)

    ax=axs[1,0]
    ax.step(to,ho['K'],'r-',lw=2.5,where='post',label='K(t)')
    ax.axhline(K_MAX,color='darkred',ls=':',lw=1.2,label=f'K_max={K_MAX}')
    for s in tsw: ax.axvline(s,color='orange',ls='--',lw=1.2,alpha=.8)
    ax2=ax.twinx()
    ax2.plot(adj['t'],adj['Sigma'],'m-',lw=1.2,alpha=.8,label='Σ(t)')
    ax2.axhline(0,color='purple',ls=':',lw=1)
    ax2.set_ylabel('Σ = ψ_θ·D_m/V',color='m',fontsize=11)
    ax.set_xlabel('Время, с'); ax.set_ylabel('K = L/D')
    ax.set_title('Управление и функция переключения')
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,fontsize=9); ax.grid(True)

    ro_o=np.array([Get_ro(h) for h in ho['h']])
    Vo=ho['V']
    qt=[(1.15e6)*(r**0.8/(0.5**0.2))*(V/7328.)**3.19 for r,V in zip(ro_o,Vo)]
    qc=[(7.845*.5)*(r/64.79)*(V/1000.)**8             for r,V in zip(ro_o,Vo)]
    qa=[h_conv*max(0.,T_atm(h)-T_wall)               for h   in ho['h']]
    q_tot=[heat_flux(h,V) for h,V in zip(ho['h'],Vo)]
    q_ball=[heat_flux(h,V) for h,V in zip(hb['h'],hb['V'])]

    ax=axs[1,1]
    ax.semilogy(to,np.maximum(q_tot ,1),'b-',lw=2,  label='q_total (ПМП)')
    ax.semilogy(tb,np.maximum(q_ball,1),'k--',lw=1.5,alpha=.6,label='q_total (баллист.)')
    ax.semilogy(to,np.maximum(qt,1),'g:',lw=1.2,label='q_turb')
    ax.semilogy(to,np.maximum(qc,1),'r:',lw=1.2,label='q_comp')
    ax.semilogy(to,np.maximum(qa,1),'m:',lw=1.2,label='q_amb')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Вт/м²')
    ax.set_title('Тепловой поток (составляющие)')
    ax.legend(fontsize=8); ax.grid(True,which='both',alpha=.3)

    ax=axs[1,2]
    ax.plot(to,ho['Q']/1e6,'b-',lw=2,label='ПМП-опт.')
    ax.plot(tb,hb['Q']/1e6,'k--',lw=1.5,alpha=.7,label='Баллист.')
    ax.set_xlabel('Время, с'); ax.set_ylabel('Q(t), МДж/м²')
    ax.set_title('Суммарный тепловой поток'); ax.legend(fontsize=10); ax.grid(True)
    plt.tight_layout()

    fig2,ax2s=plt.subplots(1,4,figsize=(19,5))
    fig2.suptitle('Сопряжённые переменные ψ(t) и гамильтониан — верификация ПМП (§11.3)',
                  fontsize=12)

    ax2s[0].plot(adj['t'],adj['pV'],'b-',lw=2)
    ax2s[0].axhline(-lambda_V,color='r',ls='--',lw=1.5,
                    label=f'ψ_V(T)=−λ_V={-lambda_V:.2f}')
    ax2s[0].set_xlabel('Время, с'); ax2s[0].set_ylabel(r'$\psi_V$')
    ax2s[0].set_title(r'$\psi_V(t)$'); ax2s[0].legend(fontsize=10); ax2s[0].grid(True)

    ax2s[1].plot(adj['t'],adj['pt'],'g-',lw=2)
    ax2s[1].axhline(0,color='r',ls='--',lw=1.5,label=r'$\psi_\theta(T)=0$')
    ax2s[1].set_xlabel('Время, с'); ax2s[1].set_ylabel(r'$\psi_\theta$')
    ax2s[1].set_title(r'$\psi_\theta(t)$'); ax2s[1].legend(fontsize=10); ax2s[1].grid(True)

    ax2s[2].plot(adj['t'],adj['pR']*1e6,'r-',lw=2)
    ax2s[2].axhline(0,color='b',ls='--',lw=1.5,label=r'$\psi_R(T)=0$')
    ax2s[2].set_xlabel('Время, с'); ax2s[2].set_ylabel(r'$\psi_R\times10^6$')
    ax2s[2].set_title(r'$\psi_R(t)$'); ax2s[2].legend(fontsize=10); ax2s[2].grid(True)

    Vi_=interp1d(to,ho['V'],fill_value='extrapolate')
    ti_=interp1d(to,ho['theta_deg']*DEG,fill_value='extrapolate')
    Ri_=interp1d(to,ho['h']+Rb,fill_value='extrapolate')
    Ki_=interp1d(to,ho['K'],kind='previous',fill_value='extrapolate')
    Harr=np.array([hamiltonian(float(Vi_(t)),float(ti_(t)),float(Ri_(t)),
                               adj['pV'][i],adj['pt'][i],adj['pR'][i],float(Ki_(t)))
                   for i,t in enumerate(adj['t'])])
    ax2s[3].plot(adj['t'],Harr,'k-',lw=2,label='H(t)')
    ax2s[3].axhline(0,color='r',ls='--',lw=1.5,label='H=0 (условие)')
    ax2s[3].set_xlabel('Время, с'); ax2s[3].set_ylabel('H')
    ax2s[3].set_title('Гамильтониан H(t)'); ax2s[3].legend(fontsize=10); ax2s[3].grid(True)
    plt.tight_layout()
    plt.show()


# [CROSS-PLATFORM] Функции-обёртки на уровне модуля — обязательно для pickle
# при spawn-контексте (Windows/macOS). С fork они тоже работают.
def _traj_opt(args):
    Ksc, tsw = args
    return integrate_traj(Ksc, tsw, save=True)

def _traj_ball(_):
    return integrate_traj([0.], [], save=True)


if __name__ == '__main__':
    # [CROSS-PLATFORM] freeze_support() нужен при упаковке в .exe (PyInstaller)
    # и безвреден при обычном запуске. Вызывать первым в __main__.
    _mp.freeze_support()

    t0 = time.time()
    Ksc, tsw, _ = solve_pmp()

    print('\nПараллельное интегрирование траекторий ...')
    # [CROSS-PLATFORM] Используем _mp_ctx() вместо хардкодированного 'fork'
    ctx = _mp_ctx()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as exe:
        fut_opt  = exe.submit(_traj_opt,  (Ksc, tsw))
        fut_ball = exe.submit(_traj_ball, None)
        V_, th_, R_, Q_, T_, ho = fut_opt.result()
        *_, hb = fut_ball.result()

    print('Обратное интегрирование сопряжённых ...')
    adj = integrate_adjoint_bwd(ho)

    print(f'Общее время: {time.time()-t0:.1f} с')
    plot_results(ho, adj, hb, Ksc, tsw)