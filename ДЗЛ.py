import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal
import matplotlib.pyplot as plt
from math import radians, degrees, sin, cos, sqrt, atan2, acos
from datetime import datetime
from scipy.special import lpmv

# Гравитационные параметры
MU_MOON = 4902.800066
MU_SUN = 1.32712440018e11  # км^3/с^2
MU_EARTH = 3.986004418e5  # км^3/с^2
R_MOON = 1737.4

# Период вращения Луны вокруг своей оси (синхронный с орбитой)
MOON_ROTATION_PERIOD = 27.3217 * 86400  # секунд

# Spherical harmonic coefficients (unnormalized)
coefficients = {
    (2, 0): (-2.033e-4, 0.0),
    (2, 1): (0.0, 0.0),
    (2, 2): (2.24e-5, 0.0),
    (3, 0): (-8.46e-6, 0.0),
    (3, 1): (2.848e-5, 5.89e-6),
    (3, 2): (4.84e-6, 1.67e-6),
    (3, 3): (1.71e-6, -0.25e-6),
    (4, 0): (1.173e-5, 0.0),
    (4, 1): (1.23e-6, 2.45e-7),
    (4, 2): (3.12e-6, 1.89e-7),
    (4, 3): (2.34e-7, -1.12e-7),
    (4, 4): (5.67e-8, 3.45e-8),
    (5, 0): (-2.388e-6, 0.0),
    (5, 1): (1.56e-6, 4.78e-7),
    (5, 2): (2.89e-7, 1.23e-7),
    (5, 3): (1.45e-7, -2.34e-8),
    (5, 4): (3.12e-8, 1.67e-8),
    (5, 5): (2.89e-8, -1.45e-8),
    (6, 0): (1.774e-5, 0.0),
    (6, 1): (2.34e-6, 3.12e-7),
    (6, 2): (4.56e-7, 2.89e-7),
    (6, 3): (1.89e-7, -3.45e-8),
    (6, 4): (4.78e-8, 2.12e-8),
    (6, 5): (3.45e-8, -1.89e-8),
    (6, 6): (2.12e-8, 1.23e-8),
}


# === Функции для учета вращения Луны ===
def get_moon_rotation_angle(t):
    """Угол поворота Луны вокруг своей оси в момент времени t"""
    return 2 * np.pi * t / MOON_ROTATION_PERIOD


def inertial_to_body_fixed(r_inertial, t):
    """Преобразование из инерциальной системы во вращающуюся (тело-фиксированную)"""
    theta = get_moon_rotation_angle(t)

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    return R @ r_inertial


def body_fixed_to_inertial(r_body, t):
    """Обратное преобразование"""
    theta = get_moon_rotation_angle(t)

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    R_inv = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    return R_inv @ r_body


# === Гравитационное ускорение с гармониками (СТАРАЯ ВЕРСИЯ) ===
def gravity_perturbation_harmonics(r):
    """Старая версия без учета вращения Луны"""
    x, y, z = r
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-3:
        return np.zeros(3)
    cos_theta = z / r_norm
    sin_theta = sqrt(1 - cos_theta ** 2)
    phi = atan2(y, x)
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    a_r = 0.0
    a_theta = 0.0
    a_phi = 0.0
    delta = 1e-8

    for n in range(2, 7):  # Уменьшим диапазон для скорости
        for m in range(0, n + 1):
            if (n, m) not in coefficients:
                continue
            C, S = coefficients[(n, m)]
            cos_mp = cos(m * phi)
            sin_mp = sin(m * phi)
            P = lpmv(m, n, cos_theta)

            V_term = MU_MOON * (R_MOON ** n / r_norm ** (n + 1)) * (C * cos_mp + S * sin_mp) * P
            a_r += (n + 1) / r_norm * V_term

            mu1 = min(1.0, max(-1.0, cos_theta + delta))
            mu2 = min(1.0, max(-1.0, cos_theta - delta))
            P1 = lpmv(m, n, mu1)
            P2 = lpmv(m, n, mu2)
            dP_dmu = (P1 - P2) / (2 * delta)
            dP_dtheta = -sin_theta * dP_dmu
            a_theta += - (1 / r_norm) * MU_MOON * (R_MOON ** n / r_norm ** (n + 1)) * (
                    C * cos_mp + S * sin_mp) * dP_dtheta

            if sin_theta > 1e-10:
                a_phi += - (1 / (r_norm * sin_theta)) * MU_MOON * (R_MOON ** n / r_norm ** (n + 1)) * (
                        -m * C * sin_mp + m * S * cos_mp) * P

    r_hat = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
    theta_hat = np.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta])
    phi_hat = np.array([-sin_phi, cos_phi, 0.0])

    return a_r * r_hat + a_theta * theta_hat + a_phi * phi_hat


# === Гравитационное ускорение с гармониками (НОВАЯ ВЕРСИЯ) ===
def gravity_perturbation_harmonics_corrected(r_inertial, t):
    """Новая версия с учетом вращения Луны"""

    r_body = inertial_to_body_fixed(r_inertial, t)
    x, y, z = r_body
    r_norm = np.linalg.norm(r_body)

    if r_norm < 1e-3:
        return np.zeros(3)

    cos_theta = z / r_norm
    sin_theta = sqrt(1 - cos_theta ** 2)
    phi = atan2(y, x)
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    a_r = 0.0
    a_theta = 0.0
    a_phi = 0.0
    delta = 1e-8

    for n in range(2, 7):  # Уменьшим диапазон для скорости
        for m in range(0, n + 1):
            if (n, m) not in coefficients:
                continue

            C, S = coefficients[(n, m)]
            cos_mp = cos(m * phi)
            sin_mp = sin(m * phi)
            P = lpmv(m, n, cos_theta)

            V_term = MU_MOON * (R_MOON ** n / r_norm ** (n + 1)) * (C * cos_mp + S * sin_mp) * P
            a_r += (n + 1) / r_norm * V_term

            mu1 = min(1.0, max(-1.0, cos_theta + delta))
            mu2 = min(1.0, max(-1.0, cos_theta - delta))
            P1 = lpmv(m, n, mu1)
            P2 = lpmv(m, n, mu2)
            dP_dmu = (P1 - P2) / (2 * delta)
            dP_dtheta = -sin_theta * dP_dmu

            a_theta += - (1 / r_norm) * MU_MOON * (R_MOON ** n / r_norm ** (n + 1)) * (
                    C * cos_mp + S * sin_mp) * dP_dtheta

            if sin_theta > 1e-10:
                a_phi += - (1 / (r_norm * sin_theta)) * MU_MOON * (R_MOON ** n / r_norm ** (n + 1)) * (
                        -m * C * sin_mp + m * S * cos_mp) * P

    r_hat = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
    theta_hat = np.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta])
    phi_hat = np.array([-sin_phi, cos_phi, 0.0])

    a_pert_body = a_r * r_hat + a_theta * theta_hat + a_phi * phi_hat
    return body_fixed_to_inertial(a_pert_body, t)


# === Конвертация вектора состояния в орбитальные элементы ===
def state_to_kepler(r, v):
    mu = MU_MOON
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    n = np.cross([0, 0, 1], h)
    n_norm = np.linalg.norm(n)
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    if e < 1e-6:
        true_anomaly = atan2(r[1], r[0])
        arg_periapsis = 0
    else:
        true_anomaly = atan2(np.dot(np.cross(e_vec, r), h) / h_norm, np.dot(e_vec, r))
        if n_norm > 1e-10:
            arg_periapsis = atan2(np.dot(np.cross(n, e_vec), h), np.dot(n, e_vec))
        else:
            arg_periapsis = atan2(e_vec[1], e_vec[0])

    a = 1 / (2 / r_norm - v_norm ** 2 / mu)
    i = acos(h[2] / h_norm)
    RAAN = atan2(n[1], n[0]) if n_norm != 0 else 0

    return {
        'a': a,
        'e': e,
        'i': degrees(i),
        'RAAN': degrees(RAAN) % 360,
        'arg_periapsis': degrees(arg_periapsis) % 360,
        'true_anomaly': degrees(true_anomaly) % 360
    }


def kepler_to_state(a, e, i, RAAN, arg_periapsis, true_anomaly):
    i = radians(i)
    RAAN = radians(RAAN)
    arg_periapsis = radians(arg_periapsis)
    true_anomaly = radians(true_anomaly)

    r = a * (1 - e ** 2) / (1 + e * cos(true_anomaly))
    x_orb = r * cos(true_anomaly)
    y_orb = r * sin(true_anomaly)
    mu = MU_MOON
    h = sqrt(mu * a * (1 - e ** 2))
    vx_orb = -mu / h * sin(true_anomaly)
    vy_orb = mu / h * (e + cos(true_anomaly))

    cos_O, sin_O = cos(RAAN), sin(RAAN)
    cos_w, sin_w = cos(arg_periapsis), sin(arg_periapsis)
    cos_i, sin_i = cos(i), sin(i)

    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i]
    ])

    r_vec = R @ np.array([x_orb, y_orb, 0])
    v_vec = R @ np.array([vx_orb, vy_orb, 0])
    return r_vec, v_vec


# === Позиции Земли и Солнца ===
def get_earth_position(t):
    angle = 2 * np.pi * t / (27.3 * 86400)
    r = 384400
    inclination = radians(5.14)
    return np.array([
        r * cos(angle),
        r * sin(angle) * cos(inclination),
        r * sin(angle) * sin(inclination)
    ])


def get_sun_position(t):
    angle = 2 * np.pi * t / (365.25 * 86400)
    r = 1.496e8
    inclination = radians(5.14)
    return np.array([
        r * cos(angle),
        r * sin(angle) * cos(inclination),
        r * sin(angle) * sin(inclination)
    ])


# === Функции ускорения ===
def acceleration(r, t, enable_J2=True, enable_earth=False, enable_sun=False, use_corrected_harmonics=False):
    a_total = -MU_MOON * r / np.linalg.norm(r) ** 3

    if enable_J2:
        if use_corrected_harmonics:
            a_total += gravity_perturbation_harmonics_corrected(r, t)
        else:
            a_total += gravity_perturbation_harmonics(r)

    if enable_earth:
        r_earth = get_earth_position(t)
        a_total += MU_EARTH * (
                (r_earth - r) / np.linalg.norm(r_earth - r) ** 3 - r_earth / np.linalg.norm(r_earth) ** 3)

    if enable_sun:
        r_sun = get_sun_position(t)
        a_total += MU_SUN * ((r_sun - r) / np.linalg.norm(r_sun - r) ** 3 - r_sun / np.linalg.norm(r_sun) ** 3)

    return a_total


def rk4_step(r, v, t, dt, enable_J2=True, enable_earth=False, enable_sun=False, use_corrected_harmonics=False):
    k1v = acceleration(r, t, enable_J2, enable_earth, enable_sun, use_corrected_harmonics)
    k1r = v

    k2v = acceleration(r + 0.5 * dt * k1r, t + 0.5 * dt, enable_J2, enable_earth, enable_sun, use_corrected_harmonics)
    k2r = v + 0.5 * dt * k1v

    k3v = acceleration(r + 0.5 * dt * k2r, t + 0.5 * dt, enable_J2, enable_earth, enable_sun, use_corrected_harmonics)
    k3r = v + 0.5 * dt * k2v

    k4v = acceleration(r + dt * k3r, t + dt, enable_J2, enable_earth, enable_sun, use_corrected_harmonics)
    k4r = v + dt * k3v

    r_new = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
    v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r_new, v_new


def simulate_orbit(a, e, i, RAAN, arg_periapsis, true_anomaly, dt, t_max,
                   enable_J2, enable_earth=False, enable_sun=False, start_epoch=0,
                   num_satellites=1, use_corrected_harmonics=False):
    positions_list = []
    elements_lists = []

    for k in range(num_satellites):
        ta_offset = k * 360.0 / num_satellites
        ta_k = true_anomaly + ta_offset
        r, v = kepler_to_state(a, e, i, RAAN, arg_periapsis, ta_k)
        positions = [r.copy()]
        elements = [state_to_kepler(r, v)]
        t = 0

        while t < t_max:
            current_t = start_epoch + t
            r, v = rk4_step(r, v, current_t, dt, enable_J2, enable_earth, enable_sun, use_corrected_harmonics)
            positions.append(r.copy())
            elements.append(state_to_kepler(r, v))
            t += dt

        positions_list.append(np.array(positions))
        elements_lists.append(elements)

    times = np.arange(0, t_max + dt, dt)[:len(positions_list[0])]
    return positions_list, times, elements_lists


# === Поток для моделирования ===
class SimulationThread(QThread):
    finished = pyqtSignal(list, np.ndarray, list)
    error = pyqtSignal(str)

    def __init__(self, params, enable_J2, enable_earth, enable_sun, start_epoch, use_corrected_harmonics):
        QThread.__init__(self)
        self.params = params
        self.enable_J2 = enable_J2
        self.enable_earth = enable_earth
        self.enable_sun = enable_sun
        self.start_epoch = start_epoch
        self.use_corrected_harmonics = use_corrected_harmonics

    def run(self):
        try:
            pos_list, times, elements_lists = simulate_orbit(
                **self.params,
                enable_J2=self.enable_J2,
                enable_earth=self.enable_earth,
                enable_sun=self.enable_sun,
                start_epoch=self.start_epoch,
                use_corrected_harmonics=self.use_corrected_harmonics
            )
            self.finished.emit(pos_list, times, elements_lists)
        except Exception as e:
            self.error.emit(str(e))


# === GUI ===
class OrbitSimulator(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Моделирование орбиты вокруг Луны")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.inputs = {}

        # Параметры ввода
        parameters = [
            ('a', 'Большая полуось (км):', '2000'),
            ('e', 'Эксцентриситет:', '0.1'),
            ('i', 'Угол наклона (°):', '30'),
            ('RAAN', 'Долгота восходящего узла (°):', '45'),
            ('arg_periapsis', 'Арг. перицентра (°):', '10'),
            ('true_anomaly', 'Истинная аномалия (°):', '0'),
            ('dt', 'Шаг (с):', '100'),
            ('t_max', 'Время моделирования (с):', '86400'),
            ('num_satellites', 'Количество спутников:', '1')
        ]

        for name, label, default in parameters:
            hbox = QHBoxLayout()
            lbl = QLabel(label)
            edit = QLineEdit(default)
            self.inputs[name] = edit
            hbox.addWidget(lbl)
            hbox.addWidget(edit)
            layout.addLayout(hbox)

        # Чекбоксы
        self.j2_checkbox = QCheckBox("Включить гармоники гравитационного поля Луны")
        self.j2_checkbox.setChecked(True)
        layout.addWidget(self.j2_checkbox)

        self.corrected_harmonics_checkbox = QCheckBox("Учитывать вращение Луны при расчете гармоник")
        self.corrected_harmonics_checkbox.setChecked(True)
        self.corrected_harmonics_checkbox.setToolTip("Более точный расчет с учетом вращения Луны вокруг своей оси")
        layout.addWidget(self.corrected_harmonics_checkbox)

        self.earth_checkbox = QCheckBox("Включить притяжение Земли")
        layout.addWidget(self.earth_checkbox)

        self.sun_checkbox = QCheckBox("Включить притяжение Солнца")
        layout.addWidget(self.sun_checkbox)

        # Кнопки
        buttons = [
            ("Показать орбиту", self.plot_orbit),
            ("Показать орбиту в 3D", self.plot_trajectories),
            ("Траектория Земли и Луны", self.plot_heliocentric_trajectories)
        ]

        for text, slot in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        # Дата
        hbox_date = QHBoxLayout()
        lbl_date = QLabel("Дата начала (ГГГГ-ММ-ДД):")
        self.date_edit = QLineEdit("2025-01-01")
        hbox_date.addWidget(lbl_date)
        hbox_date.addWidget(self.date_edit)
        layout.addLayout(hbox_date)

        self.setLayout(layout)

    def get_params(self):
        try:
            params = {}
            for name in ['a', 'e', 'i', 'RAAN', 'arg_periapsis', 'true_anomaly', 'dt', 't_max']:
                params[name] = float(self.inputs[name].text())

            params['num_satellites'] = int(self.inputs['num_satellites'].text())
            if params['num_satellites'] < 1:
                raise ValueError("Количество спутников должно быть не менее 1")

            j2 = self.j2_checkbox.isChecked()
            earth = self.earth_checkbox.isChecked()
            sun = self.sun_checkbox.isChecked()
            use_corrected = self.corrected_harmonics_checkbox.isChecked()

            start_date_str = self.date_edit.text()
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            start_epoch = (start_dt - datetime(2000, 1, 1)).total_seconds()

            return params, j2, earth, sun, start_epoch, use_corrected

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
            return None, None, None, None, None, None

    def plot_orbit(self):
        params, j2, earth, sun, start_epoch, use_corrected = self.get_params()
        if params is None:
            return

        self.simulation_thread = SimulationThread(params, j2, earth, sun, start_epoch, use_corrected)
        self.simulation_thread.finished.connect(self.on_simulation_finished)
        self.simulation_thread.error.connect(self.on_simulation_error)
        self.simulation_thread.start()

    def on_simulation_finished(self, pos_list, times, elements_lists):
        # 2D график орбиты
        plt.figure(figsize=(10, 8))
        colors = plt.cm.jet(np.linspace(0, 1, len(pos_list)))

        for i, pos in enumerate(pos_list):
            plt.plot(pos[:, 0], pos[:, 1], color=colors[i], label=f'Спутник {i + 1}', linewidth=2)

        # Луна
        moon_circle = plt.Circle((0, 0), R_MOON, color='gray', alpha=0.7, label='Луна')
        plt.gca().add_patch(moon_circle)

        plt.gca().set_aspect('equal')
        method_text = " (с учетом вращения Луны)" if self.corrected_harmonics_checkbox.isChecked() else " (старый метод)"
        plt.title(f"Орбиты вокруг Луны{method_text}")
        plt.xlabel("X (км)")
        plt.ylabel("Y (км)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Графики орбитальных параметров
        kepler_names = ['a', 'e', 'i', 'RAAN', 'arg_periapsis', 'true_anomaly']
        for param_name in kepler_names:
            plt.figure(figsize=(10, 6))
            colors = plt.cm.jet(np.linspace(0, 1, len(elements_lists)))

            for i, elements in enumerate(elements_lists):
                values = [el[param_name] for el in elements]
                plt.plot(times / 3600, values, color=colors[i], label=f'Спутник {i + 1}', linewidth=2)

            plt.xlabel("Время (часы)")
            plt.ylabel(f"{param_name}")
            plt.title(f"Эволюция параметра {param_name}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def on_simulation_error(self, error_message):
        print(f"Ошибка моделирования: {error_message}")

    def plot_trajectories(self):
        params, j2, earth, sun, start_epoch, use_corrected = self.get_params()
        if params is None:
            return

        self.simulation_thread = SimulationThread(params, j2, earth, sun, start_epoch, use_corrected)
        self.simulation_thread.finished.connect(self.on_trajectories_finished)
        self.simulation_thread.error.connect(self.on_simulation_error)
        self.simulation_thread.start()

    def on_trajectories_finished(self, pos_list, times, _):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.jet(np.linspace(0, 1, len(pos_list)))
        for i, pos in enumerate(pos_list):
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], label=f'Спутник {i + 1}', linewidth=2)

        # Луна
        ax.scatter(0, 0, 0, color='gray', s=200, label='Луна')

        ax.set_xlabel('X (км)')
        ax.set_ylabel('Y (км)')
        ax.set_zlabel('Z (км)')
        method_text = " (с учетом вращения Луны)" if self.corrected_harmonics_checkbox.isChecked() else " (старый метод)"
        ax.set_title(f'Траектории спутников{method_text}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_heliocentric_trajectories(self):
        t_max = 365.25 * 86400
        dt = 86400
        times = np.arange(0, t_max, dt)

        # Упрощенная визуализация
        earth_angle = 2 * np.pi * times / (365.25 * 86400)
        moon_angle = 2 * np.pi * times / (27.3 * 86400)

        earth_x = 1.496e8 * np.cos(earth_angle)
        earth_y = 1.496e8 * np.sin(earth_angle)

        moon_x = earth_x + 384400 * np.cos(moon_angle)
        moon_y = earth_y + 384400 * np.sin(moon_angle)

        plt.figure(figsize=(10, 8))
        plt.plot(0, 0, 'yo', markersize=15, label='Солнце')
        plt.plot(earth_x, earth_y, 'b-', label='Земля', linewidth=2)
        plt.plot(moon_x, moon_y, 'gray', label='Луна', linewidth=1)

        plt.title("Траектории в гелиоцентрической системе")
        plt.xlabel("X (км)")
        plt.ylabel("Y (км)")
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# === Запуск приложения ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OrbitSimulator()
    window.show()
    sys.exit(app.exec_())