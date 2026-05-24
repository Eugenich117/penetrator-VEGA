"""
Симуляция ориентации спутника
==============================
Динамика вращения (уравнения Эйлера) + кинематика (кватернион) +
орбитальная механика (гравитация Земли) + гравитационный градиентный момент.
ПД-регулятор по кватерниону.

Зависимости: numpy, matplotlib
  pip install numpy matplotlib
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
#  Константы
# ═══════════════════════════════════════════════════════════════════════════

MU_EARTH = 3.986004418e14   # гравитационный параметр Земли, м³/с²
RE       = 6_378_000.0      # радиус Земли, м
H_ORBIT  = 500_000.0        # высота орбиты, м

J    = np.diag([100.0, 210.0, 70.0])
Jinv = np.diag([1/100, 1/210, 1/70])

# ═══════════════════════════════════════════════════════════════════════════
#  Физика
# ═══════════════════════════════════════════════════════════════════════════

def quat_to_rot_mat(q0, q1, q2, q3):
    return np.array([
        [q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2+q0*q3),           2*(q1*q3-q0*q2)          ],
        [2*(q1*q2-q0*q3),          q0*q0-q1*q1+q2*q2-q3*q3,  2*(q2*q3+q0*q1)          ],
        [2*(q1*q3+q0*q2),          2*(q2*q3-q0*q1),           q0*q0-q1*q1-q2*q2+q3*q3 ],
    ])


def grav_grad_torque(r, q0, q1, q2, q3):
    r_norm = np.linalg.norm(r)
    e_rb   = quat_to_rot_mat(q0, q1, q2, q3) @ (r / r_norm)
    k      = 3.0 * MU_EARTH / r_norm**3
    return k * np.cross(e_rb, J @ e_rb)


def right_part(x, Ms):
    dx           = np.zeros(13)
    w            = x[0:3]
    q0,q1,q2,q3 = x[3], x[4], x[5], x[6]
    r, v         = x[7:10], x[10:13]

    # Эйлер
    mgg     = grav_grad_torque(r, q0, q1, q2, q3)
    dx[0:3] = Jinv @ (Ms + mgg - np.cross(w, J @ w))

    # Кватернион
    wx, wy, wz = w
    dx[3] = 0.5 * (-q1*wx - q2*wy - q3*wz)
    dx[4] = 0.5 * ( q0*wx + q2*wz - q3*wy)
    dx[5] = 0.5 * ( q0*wy - q1*wz + q3*wx)
    dx[6] = 0.5 * ( q0*wz + q1*wy - q2*wx)

    # Орбита
    dx[7:10]  = v
    dx[10:13] = (-MU_EARTH / np.linalg.norm(r)**3) * r
    return dx


# ═══════════════════════════════════════════════════════════════════════════
#  RK4
# ═══════════════════════════════════════════════════════════════════════════

def rk4_step(f, x, h):
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    k3 = f(x + h/2 * k2)
    k4 = f(x + h   * k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def normalize_q(x):
    x[3:7] /= np.linalg.norm(x[3:7])


# ═══════════════════════════════════════════════════════════════════════════
#  Симуляция
# ═══════════════════════════════════════════════════════════════════════════

def simulate():
    TO_RAD = math.pi / 180
    TO_DEG = 180 / math.pi

    r0  = RE + H_ORBIT
    v0  = math.sqrt(MU_EARTH / r0)
    inc = 51.6 * TO_RAD

    x = np.array([
        -0.5*TO_RAD, 1.0*TO_RAD, 0.8*TO_RAD,
         1.0, 0.0, 0.0, 0.0,
         r0,  0.0, 0.0,
         0.0, v0*math.cos(inc), v0*math.sin(inc),
    ])

    t, tk, h = 0.0, 300.0, 0.05
    Kp, Kd   = 10.0, 40.0

    data = {k: [] for k in [
        "t", "w1","w2","w3", "q0","q1","q2","q3",
        "rx","ry","rz", "vx","vy","vz",
        "alt","speed", "mgg1","mgg2","mgg3",
    ]}

    while t <= tk:
        Ms  = -Kp * x[4:7] - Kd * x[0:3]
        mgg = grav_grad_torque(x[7:10], x[3], x[4], x[5], x[6])

        x = rk4_step(lambda s: right_part(s, Ms), x, h)
        normalize_q(x)
        t += h

        data["t"].append(t)
        data["w1"].append(x[0]*TO_DEG);  data["w2"].append(x[1]*TO_DEG);  data["w3"].append(x[2]*TO_DEG)
        data["q0"].append(x[3]);         data["q1"].append(x[4])
        data["q2"].append(x[5]);         data["q3"].append(x[6])
        data["rx"].append(x[7]/1e3);     data["ry"].append(x[8]/1e3);     data["rz"].append(x[9]/1e3)
        data["vx"].append(x[10]);        data["vy"].append(x[11]);        data["vz"].append(x[12])
        data["alt"].append((np.linalg.norm(x[7:10]) - RE) / 1e3)
        data["speed"].append(np.linalg.norm(x[10:13]))
        data["mgg1"].append(mgg[0]);     data["mgg2"].append(mgg[1]);     data["mgg3"].append(mgg[2])

    return {k: np.array(v) for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Графики
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(data):
    ts = data["t"]

    charts = [
        # (ключ,        заголовок,            единица)
        ("w1",    "Угловая скорость ω₁",  "°/с"),
        ("w2",    "Угловая скорость ω₂",  "°/с"),
        ("w3",    "Угловая скорость ω₃",  "°/с"),
        ("q0",    "Кватернион q₀",         ""),
        ("q1",    "Кватернион q₁",         ""),
        ("q2",    "Кватернион q₂",         ""),
        ("q3",    "Кватернион q₃",         ""),
        ("rx",    "Положение rx",          "км"),
        ("ry",    "Положение ry",          "км"),
        ("rz",    "Положение rz",          "км"),
        ("vx",    "Скорость vx",           "м/с"),
        ("vy",    "Скорость vy",           "м/с"),
        ("vz",    "Скорость vz",           "м/с"),
        ("alt",   "Высота орбиты",         "км"),
        ("speed", "Орбитальная скорость",  "м/с"),
        ("mgg1",  "Mgg_x (гр. градиент)", "Н·м"),
        ("mgg2",  "Mgg_y (гр. градиент)", "Н·м"),
        ("mgg3",  "Mgg_z (гр. градиент)", "Н·м"),
    ]

    # 18 графиков на 6 строк × 3 столбца
    fig, axes = plt.subplots(6, 3, figsize=(18, 20))
    fig.suptitle("Симуляция ориентации спутника", fontsize=16, fontweight="bold")

    colors = [
        "#f38ba8","#a6e3a1","#89b4fa",
        "#f9e2af","#f38ba8","#a6e3a1",
        "#89b4fa","#fab387","#cba6f7",
        "#94e2d5","#fab387","#cba6f7",
        "#94e2d5","#f9e2af","#74c7ec",
        "#f38ba8","#a6e3a1","#89b4fa",
    ]

    for ax, (key, title, unit), color in zip(axes.flat, charts, colors):
        ax.plot(ts, data[key], color=color, linewidth=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Время, с", fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Симуляция...")
    data = simulate()
    print(f"Готово: {len(data['t'])} точек, t_конец={data['t'][-1]:.1f} с")
    plot_all(data)