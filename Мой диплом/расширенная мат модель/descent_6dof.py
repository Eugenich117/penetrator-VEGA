"""
descent_6dof.py — спуск зонда-пенетратора в атмосфере Венеры как ТВЁРДОГО ТЕЛА
(6 степеней свободы: 3 — движение центра масс, 3 — вращение вокруг центра масс)
с учётом вращения планеты, стохастического ветра (venus_wind.py) и закрутки аппарата.

СИСТЕМЫ КООРДИНАТ
-----------------
I — невращающаяся планетоцентрическая (инерциальная): ось z — к северному полюсу Венеры
    (по МАС), ось x — на долготу 0 в момент t = 0. В ней интегрируется всё движение,
    поэтому кориолисово и центробежное ускорения учитываются автоматически.
P — связанная с планетой: поворот I вокруг z на угол Ω·t, Ω = OMEGA_VENUS = −2,9924e-7 рад/с
    (вращение обратное). В ней заданы долгота, атмосфера и ветер.
B — связанная с аппаратом: ось x_b — продольная ось к носку, y_b, z_b — поперечные.
    Ориентация B относительно I — кватернион q (B -> I).

СОСТОЯНИЕ (13 чисел): r_I (3), v_I (3), q (4), ω_B (3)

    dr/dt = v
    dv/dt = −μ r/|r|³ + C_IB·F_B / m                      (тяготение + аэродинамика)
    dq/dt = ½ q ⊗ (0, ω_B)                                 (кинематика кватерниона)
    I·dω/dt = M_B − ω × (I·ω),  I = diag(Ix, It, It)        (уравнения Эйлера)

СКОРОСТЬ ОТНОСИТЕЛЬНО ВОЗДУХА
    v_air = v − Ω×r − W,   W = u·e_E + v·e_N + w·e_U      (атмосфера вращается с планетой,
                                                             ветер — относительно поверхности)
    v_air,B = C_BI·v_air,   α — пространственный угол атаки между x_b и v_air,B.

АЭРОДИНАМИЧЕСКИЕ СИЛЫ (в осях B, q = ½ρ|v_air|²)
    F_B = −½ ρ |v_air| S · (C_A·v_x, C_Nα·v_y, C_Nα·v_z)
    т.е. осевая сила C_A·cosα и нормальная C_Nα·sinα, направленная против поперечной
    составляющей v_air. При α = 0 это в точности сила сопротивления 3-DOF модели
    (descent_core.rhs), C_A = Cx(M) — ваша таблица/функция cx_func.
    C_Nα = cn_alpha_ratio · C_A. По ньютоновской теории для конуса с полууглом θ:
    C_A = 2 sin²θ, C_Nα = 2 cos²θ, отношение ctg²θ. Ваше Cx = 1,52 соответствует θ ≈ 60,7°,
    ctg²θ = 0,317 (значение по умолчанию).

МОМЕНТЫ ОТНОСИТЕЛЬНО ЦЕНТРА МАСС
    Аэродинамическая сила приложена в центре давления на оси аппарата. Центр масс
    смещён относительно ЦД на x_st вперёд (запас статической устойчивости, xd в
    скрипте перебора Qk) и на y_cm вбок (gamma_ в том же скрипте). Плечо r = (−x_st, −y_cm, 0):
        M_aero = r × F_B
    Демпфирующие моменты:
        M_x = q S d · C_lp · ω_x d/(2V),   M_y,z = q S d · C_mq · ω_y,z d/(2V).
    Смещение ЦМ даёт балансировочный угол атаки α_бал ≈ y_cm·C_A/(x_st·C_Nα) и боковую
    силу. Без вращения она всё время направлена в одну сторону -> систематический увод.
    При закрутке направление силы вращается вместе с аппаратом и в среднем гасится.
    Колебания по углу атаки затухают, если C_Lα − m d²/(2I_t)·C_mq > 0 (Body.stability_criterion).
    Опасный режим — РЕЗОНАНС: собственная частота колебаний по углу атаки
        ω_n = √(q S C_Nα x_st / It)
    растёт с напором; когда ω_n/√(1 − Ix/It) проходит через скорость закрутки p,
    балансировочный угол усиливается в 1/|1 − p²(1 − Ix/It)/ω_n²| раз.

ЗАКРУТКА: только начальная (как у пенетраторов «Марс-96»: 1,25 об/с при отделении от
орбитального аппарата, «Новости космонавтики», 1996, № 22–23). В атмосфере она затухает
от демпфирования по крену C_lp.

ПАРАМЕТРЫ АППАРАТА (класс Body) — оценки, см. комментарии; замените своими данными.

ИНТЕГРИРОВАНИЕ: Рунге–Кутта 4-го порядка с переменным шагом dt = c / ω_хар, где
ω_хар — наибольшая из угловой скорости аппарата, ω_n и 1/τ торможения. Плотность и
скорость звука считаются на каждой стадии по вашей таблице (descent_core.Get_ro,
v_sound), предвычисленной на сетке 10 м.
"""

from __future__ import annotations

import math as m
from dataclasses import dataclass

import numpy as np

import descent_core as dc
from descent_core import Rb, g0, cToRad, cToDeg, OMEGA_VENUS

MU = g0 * Rb * Rb                       # то же поле тяготения, что и в descent_core

# ---------------------------------------------------------------------------
# Атмосфера: ваша таблица (Get_ro, v_sound), предвычисленная на сетке 10 м,
# внутри шага — линейная интерполяция ln ρ и a (быстро и гладко)
# ---------------------------------------------------------------------------
_DH = 10.0
_HT = np.arange(0.0, 130_000.0 + _DH, _DH)
_LNRO = np.log([dc.Get_ro(h) for h in _HT]).tolist()
_ASND = [dc.v_sound(h) for h in _HT]
_NT = len(_LNRO)


def atmosphere(h):
    """(ρ, a) на высоте h, м."""
    s = h / _DH
    if s <= 0.0:
        return m.exp(_LNRO[0]), _ASND[0]
    if s >= _NT - 1:
        return m.exp(_LNRO[-1]), _ASND[-1]
    i = int(s)
    f = s - i
    return (m.exp(_LNRO[i] + f * (_LNRO[i + 1] - _LNRO[i])),
            _ASND[i] + f * (_ASND[i + 1] - _ASND[i]))


# ---------------------------------------------------------------------------
# Параметры аппарата
# ---------------------------------------------------------------------------
def inertia_estimate(mass=120.0, m_aft=30.0, L_body=2.1, d_body=0.17, d_aft=0.8, L_aft=0.6):
    """
    Оценка моментов инерции для компоновки типа пенетратора «Марс-96»:
    тонкий корпус (сплошной цилиндр L_body × d_body, масса mass − m_aft) и хвостовой
    конический обтекатель-тормоз (тонкая коническая оболочка от d_body до d_aft длиной L_aft
    в хвосте, масса m_aft). Размеры «Марс-96»: длина 2,1 м, корпус 0,14–0,17 м,
    хвостовой конус 0,8 м, масса 123 кг («Новости космонавтики», 1996, № 22–23).
    Возвращает (Ix, It относительно ЦМ, x_cm от носка).
    """
    m_c = mass - m_aft
    r_c = d_body / 2
    x_c = L_body / 2
    # оболочка: кольца, масса кольца ∝ радиусу (площадь боковой поверхности)
    xs = np.linspace(L_body - L_aft, L_body, 400)
    rs = np.linspace(d_body / 2, d_aft / 2, 400)
    w = rs / rs.sum() * m_aft
    x_cm = (m_c * x_c + (w * xs).sum()) / mass
    Ix = 0.5 * m_c * r_c ** 2 + (w * rs ** 2).sum()
    It = (m_c * (3 * r_c ** 2 + L_body ** 2) / 12 + m_c * (x_c - x_cm) ** 2
          + (w * (rs ** 2 / 2 + (xs - x_cm) ** 2)).sum())
    return float(Ix), float(It), float(x_cm)


_IX0, _IT0, _XCM0 = inertia_estimate()


@dataclass
class Body:
    mass: float = 120.0                 # кг (ваше значение; «Марс-96» — 123 кг)
    d_ref: float = 0.34                  # м, диаметр миделя: S = πd²/4, характерная длина моментов
    Ix: float = _IX0                    # кг·м², продольный момент инерции (оценка ≈ 2,8)
    It: float = _IT0                    # кг·м², поперечный относительно ЦМ (оценка ≈ 52)
    x_st: float = 0.006                  # м, ЦД позади ЦМ (xd из скрипта перебора Qk)
    y_cm: float = 0.0009                # м, поперечное смещение ЦМ (gamma_ оттуда же); 0 — симметричный
    cn_alpha_ratio: float = 0.317       # C_Nα / C_A, ньютоновская оценка для конуса θ ≈ 60,7°
    cmq: float = -1.5                   # C_mq + C_mα̇, демпфирование тангажа. ОЦЕНКА для компоновки с хвостовым
                                        #  коническим стабилизатором (как «Марс-96»): ≈ −2·C_Nα,ст·(l_ст/d)² ≈ −1,5…−2.
                                        #  У затупленного конуса без стабилизатора −0,1…−0,4, и тогда аппарат на
                                        #  дозвуке динамически неустойчив: см. stability_criterion()
    clp: float = -0.01                  # C_lp, демпфирование крена (трение; гладкое тело — мало)

    @property
    def S(self):
        return m.pi * self.d_ref ** 2 / 4

    def stability_criterion(self, CA):
        """
        Динамическая устойчивость колебаний по углу атаки (линейная теория, плоское движение):
            α'' + c·α' + ω_n²·α = 0,   c = (ρVS / 2m)·[C_Lα − m d²/(2 I_t)·C_mq],
        C_Lα = C_Nα − C_A (наклон подъёмной силы; у затупленных тел отрицателен).
        Колебания затухают при > 0. Возвращает значение скобки при данном C_A.
        """
        return (self.cn_alpha_ratio - 1.0) * CA - self.mass * self.d_ref ** 2 / (2 * self.It) * self.cmq


@dataclass
class Params6:
    V0: float                           # м/с, скорость входа
    tetta0: float                       # рад, траекторный угол входа (< 0)
    azimuth: float                      # рад, азимут входа от севера по часовой
    lat0: float = 0.0
    lon0: float = 0.0
    h0: float = 125_000.0
    entry_frame: str = "inertial"       # V0, θ0, ψ0 — "inertial" или "relative" (к поверхности)
    omega_planet: float = OMEGA_VENUS
    spin0: float = 1.25 * 2 * m.pi      # рад/с, начальная закрутка (Марс-96: 1,25 об/с)
    alpha0: float = 0.0                 # рад, начальный угол между осью и скоростью
    alpha0_phase: float = 0.0           # рад, плоскость начального угла
    roll0: float = 0.0                  # рад, начальный угол крена (в Монте-Карло — случайный 0…2π:
                                        #  от него зависит, куда направлена боковая сила от смещения ЦМ)
    rate0: tuple = (0.0, 0.0)           # рад/с, начальные поперечные угловые скорости (ω_y, ω_z)
    dt_min: float = 1e-3
    dt_max: float = 0.1
    step_k: float = 0.1                 # dt = step_k / ω_хар (≤ 0,1 рад за шаг: нутация не затухает численно)
    save_dt: float = 0.1                # с, шаг записи результатов


# ---------------------------------------------------------------------------
# Кватернионы (q0 — скалярная часть), q: B -> I
# ---------------------------------------------------------------------------
def quat_to_dcm(q0, q1, q2, q3):
    """Матрица C_IB (столбцы — орты осей B в системе I)."""
    return ((1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)),
            (2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)),
            (2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)))


def dcm_to_quat(C):
    C = np.asarray(C)
    tr = np.trace(C)
    if tr > 0:
        s = m.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (C[2, 1] - C[1, 2]) / s, (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(C)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = m.sqrt(1.0 + C[i, i] - C[j, j] - C[k, k]) * 2
        qv = [0.0, 0.0, 0.0]
        qv[i] = 0.25 * s
        qv[j] = (C[j, i] + C[i, j]) / s
        qv[k] = (C[k, i] + C[i, k]) / s
        q = ((C[k, j] - C[j, k]) / s, *qv)
    n = m.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q)


def enu_axes(x, y, z):
    """Орты e_E, e_N, e_U в системе I для точки r = (x, y, z); sinφ, cosφ."""
    r = m.sqrt(x * x + y * y + z * z)
    rxy = m.sqrt(x * x + y * y)
    cl, sl = x / rxy, y / rxy
    sp, cp = z / r, rxy / r
    return (-sl, cl, 0.0), (-sp * cl, -sp * sl, cp), (cp * cl, cp * sl, sp), r


# ---------------------------------------------------------------------------
# Правая часть
# ---------------------------------------------------------------------------
def rhs6(t, s, body: Body, cx_func, wind, Om, out=None):
    x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = s
    eE, eN, eU, r = enu_axes(x, y, z)
    h = r - Rb
    rho, a_snd = atmosphere(h)
    if wind is not None:
        uw, vw_, ww = wind(h)
    else:
        uw = vw_ = ww = 0.0
    # скорость относительно воздуха в I: v − Ω×r − W
    ax_ = vx + Om * y - (uw * eE[0] + vw_ * eN[0] + ww * eU[0])
    ay_ = vy - Om * x - (uw * eE[1] + vw_ * eN[1] + ww * eU[1])
    az_ = vz - (vw_ * eN[2] + ww * eU[2])
    C = quat_to_dcm(q0, q1, q2, q3)
    # в осях B: v_B = C_IBᵀ v_I
    bx = C[0][0] * ax_ + C[1][0] * ay_ + C[2][0] * az_
    by = C[0][1] * ax_ + C[1][1] * ay_ + C[2][1] * az_
    bz = C[0][2] * ax_ + C[1][2] * ay_ + C[2][2] * az_
    Va = m.sqrt(bx * bx + by * by + bz * bz)
    Mach = Va / a_snd
    CA = cx_func(Mach, h)
    CNa = body.cn_alpha_ratio * CA
    S, d = body.S, body.d_ref
    kq = 0.5 * rho * Va * S                       # F = −kq·(CA vx, CNa vy, CNa vz)
    Fx, Fy, Fz = -kq * CA * bx, -kq * CNa * by, -kq * CNa * bz
    # моменты: плечо ЦД относительно ЦМ r = (−x_st, −y_cm, 0)
    xs, yc = body.x_st, body.y_cm
    Mx = -yc * Fz
    My = xs * Fz
    Mz = -xs * Fy + yc * Fx
    if Va > 1e-6:
        qd = 0.25 * rho * Va * S * d * d          # q S d · d/(2V)
        Mx += qd * body.clp * wx
        My += qd * body.cmq * wy
        Mz += qd * body.cmq * wz
    Ix, It = body.Ix, body.It
    dwx = Mx / Ix
    dwy = (My - (Ix - It) * wx * wz) / It
    dwz = (Mz - (It - Ix) * wx * wy) / It
    # поступательное движение в I
    mass = body.mass
    gk = -MU / (r * r * r)
    dvx = gk * x + (C[0][0] * Fx + C[0][1] * Fy + C[0][2] * Fz) / mass
    dvy = gk * y + (C[1][0] * Fx + C[1][1] * Fy + C[1][2] * Fz) / mass
    dvz = gk * z + (C[2][0] * Fx + C[2][1] * Fy + C[2][2] * Fz) / mass
    # кинематика кватерниона: q̇ = ½ q ⊗ (0, ω)
    dq0 = 0.5 * (-q1 * wx - q2 * wy - q3 * wz)
    dq1 = 0.5 * (q0 * wx + q2 * wz - q3 * wy)
    dq2 = 0.5 * (q0 * wy + q3 * wx - q1 * wz)
    dq3 = 0.5 * (q0 * wz + q1 * wy - q2 * wx)
    if out is not None:
        q_dyn = 0.5 * rho * Va * Va
        out.update(h=h, rho=rho, Va=Va, Mach=Mach, CA=CA, q=q_dyn, bx=bx, by=by, bz=bz,
                   F=(Fx, Fy, Fz), C=C, eE=eE, eN=eN, eU=eU, r=r, W=(uw, vw_, ww),
                   omega_n=m.sqrt(max(q_dyn * S * CNa * xs / It, 0.0)),
                   k_drag=kq * CA / mass)
    return (vx, vy, vz, dvx, dvy, dvz, dq0, dq1, dq2, dq3, dwx, dwy, dwz)


def rk4(t, s, dt, *args):
    k1 = rhs6(t, s, *args)
    k2 = rhs6(t + 0.5 * dt, tuple(a + 0.5 * dt * b for a, b in zip(s, k1)), *args)
    k3 = rhs6(t + 0.5 * dt, tuple(a + 0.5 * dt * b for a, b in zip(s, k2)), *args)
    k4 = rhs6(t + dt, tuple(a + dt * b for a, b in zip(s, k3)), *args)
    s = [a + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
    n = m.sqrt(s[6] ** 2 + s[7] ** 2 + s[8] ** 2 + s[9] ** 2)   # нормировка кватерниона
    s[6:10] = [c / n for c in s[6:10]]
    return tuple(s)


# ---------------------------------------------------------------------------
# Начальные условия
# ---------------------------------------------------------------------------
def initial_state(p: Params6):
    r0 = Rb + p.h0
    cp, sp = m.cos(p.lat0), m.sin(p.lat0)
    cl, sl = m.cos(p.lon0), m.sin(p.lon0)
    x, y, z = r0 * cp * cl, r0 * cp * sl, r0 * sp
    eE, eN, eU, _ = enu_axes(x, y, z)
    ct = m.cos(p.tetta0)
    vE, vN, vU = p.V0 * ct * m.sin(p.azimuth), p.V0 * ct * m.cos(p.azimuth), p.V0 * m.sin(p.tetta0)
    v = np.array(eE) * vE + np.array(eN) * vN + np.array(eU) * vU
    if p.entry_frame == "relative":            # V_ин = V_отн + Ω×r
        v = v + p.omega_planet * np.array([-y, x, 0.0])
    elif p.entry_frame != "inertial":
        raise ValueError(f"entry_frame: {p.entry_frame!r}")
    # ориентация: ось x_b вдоль инерциальной скорости (закрученный аппарат сохраняет
    # направление оси в пространстве на пассивном участке), отклонённая на alpha0
    e1 = v / np.linalg.norm(v)
    e2 = np.cross(e1, eU)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    ca, sa = m.cos(p.alpha0), m.sin(p.alpha0)
    xb = ca * e1 + sa * (m.cos(p.alpha0_phase) * e2 + m.sin(p.alpha0_phase) * e3)
    yb = np.cross(e3 if abs(np.dot(xb, e3)) < 0.9 else e2, xb)
    yb /= np.linalg.norm(yb)
    zb = np.cross(xb, yb)
    cr, sr = m.cos(p.roll0), m.sin(p.roll0)
    yb, zb = cr * yb + sr * zb, -sr * yb + cr * zb
    q = dcm_to_quat(np.column_stack([xb, yb, zb]))
    return (x, y, z, *v.tolist(), *q, p.spin0, *p.rate0)


# ---------------------------------------------------------------------------
# Интегрирование
# ---------------------------------------------------------------------------
REC_KEYS = ("T", "H", "lat", "lon", "V", "Va", "TETTA", "PSI", "napor", "nx", "CX", "MAH",
            "alpha", "p_rps", "omega_t", "omega_n", "omega_res", "Wu", "Wv", "Ww", "Wx", "Wz",
            "lift_dir")


def _record(rec, t, s, p, body, cx_func, wind):
    o = {}
    rhs6(t, s, body, cx_func, wind, p.omega_planet, o)
    x, y, z, vx, vy, vz = s[:6]
    Om = p.omega_planet
    vr = (vx + Om * y, vy - Om * x, vz)            # скорость относительно поверхности, в I
    eE, eN, eU = o["eE"], o["eN"], o["eU"]
    VE = sum(a * b for a, b in zip(vr, eE))
    VN = sum(a * b for a, b in zip(vr, eN))
    VU = sum(a * b for a, b in zip(vr, eU))
    V, th, psi = dc.flight_angles(VE, VN, VU)
    lat = m.asin(z / o["r"])
    lon = m.atan2(y, x) - Om * t                    # долгота в системе, связанной с планетой
    Va = o["Va"]
    alpha = m.acos(max(-1.0, min(1.0, o["bx"] / Va))) if Va > 0 else 0.0
    F = o["F"]
    C = o["C"]
    FI = [C[i][0] * F[0] + C[i][1] * F[1] + C[i][2] * F[2] for i in range(3)]
    gR = MU / o["r"] ** 2
    nx = m.sqrt(F[0] ** 2 + F[1] ** 2 + F[2] ** 2) / (body.mass * gR)
    # направление боковой силы (составляющей, перпендикулярной v_air) в плоскости E–N, град
    var = [vr[i] - (o["W"][0] * eE[i] + o["W"][1] * eN[i] + o["W"][2] * eU[i]) for i in range(3)]
    vn = m.sqrt(sum(c * c for c in var))
    Fpar = sum(a * b for a, b in zip(FI, var)) / vn
    Fl = [FI[i] - Fpar * var[i] / vn for i in range(3)]
    lift_dir = m.degrees(m.atan2(sum(a * b for a, b in zip(Fl, eE)), sum(a * b for a, b in zip(Fl, eN))))
    uw, vw_, ww = o["W"]
    rt = 1.0 - body.Ix / body.It
    for k, v in zip(REC_KEYS, (t, o["h"], lat, lon, V, Va, th * cToDeg, psi * cToDeg, o["q"], nx,
                               o["CA"], o["Mach"], alpha * cToDeg, s[10] / (2 * m.pi),
                               m.hypot(s[11], s[12]), o["omega_n"],
                               o["omega_n"] / m.sqrt(rt) if rt > 0 else float("nan"),
                               uw, vw_, ww, uw * m.sin(psi) + vw_ * m.cos(psi),
                               uw * m.cos(psi) - vw_ * m.sin(psi), lift_dir)):
        rec[k].append(v)
    return o


def simulate6(p: Params6, body: Body, cx_func, wind=None, tag=None, t_max=20_000.0):
    s = initial_state(p)
    t = 0.0
    rec = {k: [] for k in REC_KEYS}
    t_next = 0.0
    n_steps = 0
    o = {}
    while True:
        if t >= t_next:
            o = _record(rec, t, s, p, body, cx_func, wind)
            t_next += p.save_dt
        else:
            o = {}
            rhs6(t, s, body, cx_func, wind, p.omega_planet, o)
        w_char = max(m.sqrt(s[10] ** 2 + s[11] ** 2 + s[12] ** 2), o["omega_n"], o["k_drag"], 1e-3)
        dt = min(p.dt_max, max(p.dt_min, p.step_k / w_char))
        s_prev, t_prev = s, t
        s = rk4(t, s, dt, body, cx_func, wind, p.omega_planet)
        t += dt
        n_steps += 1
        r = m.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
        if r < Rb or t > t_max:
            break
    # точка падения: линейная интерполяция внутри последнего шага
    r_prev = m.sqrt(s_prev[0] ** 2 + s_prev[1] ** 2 + s_prev[2] ** 2)
    f = (r_prev - Rb) / (r_prev - r)
    sl = tuple(a + f * (b - a) for a, b in zip(s_prev, s))
    tl = t_prev + f * (t - t_prev)
    land_rec = {k: [] for k in REC_KEYS}
    _record(land_rec, tl, sl, p, body, cx_func, wind)
    out = {k: np.asarray(v) for k, v in rec.items()}
    frame = dc.track_frame(p.lat0, p.lon0, p.azimuth)
    out["L"], out["Z"] = dc.downrange_crossrange(out["lat"], out["lon"], frame)
    Ll, Zl = dc.downrange_crossrange(land_rec["lat"][0], land_rec["lon"][0], frame)
    out["landing"] = dict(V=land_rec["V"][0], tetta=land_rec["TETTA"][0] * cToRad,
                          psi=land_rec["PSI"][0] * cToRad, L=float(Ll), Z=float(Zl),
                          lat=land_rec["lat"][0], lon=land_rec["lon"][0], t=tl,
                          alpha=land_rec["alpha"][0], p_rps=land_rec["p_rps"][0])
    out["acceleration"] = np.gradient(out["V"], out["T"]) if out["T"].size > 1 else np.zeros(1)
    out["PX"] = body.mass / (out["CX"] * body.S)
    out["tag"], out["n_steps"] = tag, n_steps
    out["V0"], out["tetta0"], out["azimuth"] = p.V0, p.tetta0, p.azimuth
    return out
