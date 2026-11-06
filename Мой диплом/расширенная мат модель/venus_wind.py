"""
venus_wind.py — стохастическая модель ветра атмосферы Венеры для статистического
(Монте-Карло) моделирования спуска зонда-пенетратора.

ИДЕЯ МОДЕЛИ
-----------
1. Ветер — случайная функция ВЫСОТЫ, а не времени (гипотеза «замороженного» поля).
   Одна реализация генерируется на весь спуск заранее:

       W(h) = W̄(h) + W'(h)

   Обоснование: спуск длится ~50–60 мин, а крупномасштабное поле суперротации
   меняется за сутки (облачный слой обращается вокруг планеты за ~4 земных суток), т.е. за время
   спуска оно практически неизменно. Для мелкомасштабной конвекции в слое 50–55 км
   (время жизни ячеек ~ десятки минут) это допущение — приближение; её вклад
   в снос мал, а на нагрузки влияет через сдвиг ветра, который модель воспроизводит.

2. Компоненты в местной системе ENU (восток, север, вверх):
       u — зональная (суперротация: ветер почти строго на запад, u < 0),
       v — меридиональная: среднее v̄(h, φ) (ячейка Хэдли, к полюсу у верхней
           границы облаков) + случайная часть,
       w — вертикальная (заметна только в конвективном слое 50–55 км).

3. Корреляция по высоте — формирующий фильтр (подход статистической динамики ЛА,
   Бобронников В.Т. и др., МАИ): стандартный гауссов процесс ξ(h) — выход
   линейного фильтра, на вход которого подан белый шум.
       порядок 1:  H(s) = √(2α) / (s + α)        K(Δh) = exp(-α|Δh|)
       порядок 2:  H(s) = √(4α³) / (s + α)²      K(Δh) = (1 + α|Δh|)·exp(-α|Δh|)
   α = 1 / L_corr. Фильтр 2-го порядка даёт дифференцируемую реализацию.
   Дискретизация точная (матричная экспонента), коэффициенты фильтра
   вычисляются ОДИН раз при создании модели (класс ShapingFilter).

4. ДВУХМАСШТАБНАЯ СТРУКТУРА (two_scale=True, по умолчанию):
       ξ(h) = √(1-β(h))·ξ_L(h) + √β(h)·ξ_S(h),
   ξ_L — крупномасштабный процесс (L_L ~ 5–15 км: различия между профилями
   разных зондов, определяют разброс сноса), ξ_S — мелкомасштабная турбулентность
   (L_S ~ 1–5 км: определяет сдвиг ветра). ξ_L и ξ_S независимы, оба N(0,1),
   поэтому ξ — тоже N(0,1) на каждой высоте, а корреляционная функция — смесь
       K(Δh) = (1-β)·K_L(Δh) + β·K_S(Δh),    β = (σ_турб / σ)².
   two_scale=False — исходная одномасштабная модель (только ξ_S); при тех же
   seed даёт те же реализации, что и предыдущая версия файла.

5. Зональная скорость U = -u ≥ 0 имеет на каждой высоте закон Вейбулла
       F(U) = 1 - exp(-(U/λ)^k),     λ(h) = Ū(h) / Γ(1 + 1/k(h)),
   получаемый безынерционным преобразованием  U = F_W⁻¹(Φ(ξ)).
   Так как ξ ~ N(0,1) в любом сечении (в т.ч. в двухмасштабной модели),
   закон распределения U в каждом сечении — ТОЧНО вейбулловский.

6. u, v, w независимы между собой.

ОГРАНИЧЕНИЯ (назвать в списке допущений)
  * Вейбулл задан для зональной компоненты, а не для модуля горизонтальной
    скорости; выше ~10 км это одно и то же (|v| ≪ U), ниже различие есть, но
    вклад слоя 0–10 км в снос — единицы км.
  * Не моделируется течение «подсолнечная — антисолнечная точка» выше ~90 км
    (зависит от местного солнечного времени). На траекторию влияет слабо:
    плотность мала, скорость аппарата ~10 км/с.
  * Одинаковые L_S, L_L для u и v; w — только мелкомасштабная.

ЗНАЧЕНИЯ ПАРАМЕТРОВ (таблица PROFILE) — инженерная оценка по данным Pioneer Venus,
ВеГа, Venus Express. Столбцы L_L, σ_турб, V_m добавлены в этой версии и являются
ПРЕДПОЛОЖЕНИЯМИ — их нужно откалибровать (L_L сильнее всего влияет на эллипс
рассеивания). Код от конкретных чисел не зависит.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gamma as gamma_fn, ndtr


# ---------------------------------------------------------------------------
# Параметры профиля
#   h      — высота, км
#   U      — Ū, средняя зональная скорость (на запад), м/с, на экваторе
#   k      — параметр формы Вейбулла для U
#   sv, sw — полные СКО меридиональной и вертикальной скорости, м/с
#   Ls     — масштаб корреляции мелкомасштабной (турбулентной) части, км
#   Ll     — масштаб корреляции крупномасштабной части, км          [предположение]
#   st     — СКО турбулентной части горизонтального ветра, м/с      [предположение]
#   Vm     — средняя меридиональная скорость к полюсу при |φ| = 45°, м/с [предположение]
# ---------------------------------------------------------------------------
PROFILE_COLUMNS = ("h", "U", "k", "sv", "sw", "Ls", "Ll", "st", "Vm")
PROFILE = np.array([
    #  h      U     k     sv    sw    Ls    Ll    st    Vm
    [  0.0,   0.5,  2.0,  0.5, 0.10,  1.0,  5.0, 0.2,  0.0],
    [  2.0,   1.5,  2.2,  0.8, 0.10,  1.5,  6.0, 0.3,  0.0],
    [  6.0,   3.5,  2.5,  1.0, 0.15,  2.0,  7.0, 0.5,  0.0],
    [ 10.0,   7.0,  3.0,  1.5, 0.20,  2.0,  8.0, 0.8,  0.0],
    [ 15.0,  13.0,  3.5,  2.0, 0.25,  2.0, 10.0, 1.0,  0.0],   # слой сильного сдвига ~15 км
    [ 20.0,  20.0,  4.0,  2.0, 0.30,  2.5, 12.0, 1.0,  0.0],
    [ 25.0,  26.0,  4.5,  2.0, 0.40,  3.0, 12.0, 1.0,  0.0],
    [ 30.0,  32.0,  5.0,  2.5, 0.40,  3.0, 12.0, 1.0,  0.0],
    [ 35.0,  38.0,  5.5,  2.5, 0.40,  3.0, 12.0, 1.0,  0.0],
    [ 40.0,  45.0,  6.0,  3.0, 0.50,  3.0, 12.0, 1.0,  0.0],
    [ 45.0,  55.0,  6.0,  3.0, 0.60,  2.0, 12.0, 1.5,  0.0],   # слой сдвига ~45 км
    [ 50.0,  62.0,  6.0,  4.0, 1.20,  1.0, 12.0, 2.5,  0.0],   # конвективный слой 50–55 км
    [ 55.0,  68.0,  6.0,  5.0, 1.20,  1.0, 12.0, 2.5,  2.0],
    [ 60.0,  85.0,  7.0,  6.0, 0.80,  1.5, 12.0, 2.0,  5.0],   # слой сдвига ~60 км
    [ 65.0, 100.0,  8.0,  8.0, 0.50,  2.5, 12.0, 1.5,  8.0],   # верхняя граница облаков
    [ 70.0, 100.0,  8.0, 10.0, 0.50,  3.0, 12.0, 1.5,  8.0],
    [ 75.0,  92.0,  7.0, 10.0, 0.50,  3.0, 12.0, 1.5,  5.0],
    [ 80.0,  78.0,  6.0, 10.0, 0.50,  4.0, 12.0, 1.5,  2.0],
    [ 85.0,  62.0,  5.0, 10.0, 0.50,  4.0, 12.0, 1.5,  0.0],
    [ 90.0,  48.0,  4.0, 10.0, 0.50,  5.0, 12.0, 1.5,  0.0],
    [ 95.0,  32.0,  3.5,  8.0, 0.50,  5.0, 12.0, 1.5,  0.0],
    [100.0,  20.0,  3.0,  6.0, 0.50,  5.0, 12.0, 1.5,  0.0],
    [110.0,  10.0,  2.5,  4.0, 0.50,  5.0, 15.0, 1.0,  0.0],
    [130.0,   5.0,  2.0,  2.0, 0.50,  5.0, 15.0, 1.0,  0.0],
])


def latitude_factor(lat_deg: float) -> float:
    """Ослабление суперротации с широтой: ~постоянна до |φ|≈50°, далее линейно до 0 к полюсу."""
    a = abs(lat_deg)
    return 1.0 if a <= 50.0 else max(0.0, (90.0 - a) / 40.0)


def meridional_factor(lat_deg: float) -> float:
    """Широтная форма среднего меридионального течения: sin 2φ (0 на экваторе и полюсе,
    максимум при |φ| = 45°, знак — к полюсу своего полушария)."""
    return math.sin(2.0 * math.radians(lat_deg))


def weibull_scale(mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """λ из условия M[U] = λ·Γ(1 + 1/k)."""
    return mean / gamma_fn(1.0 + 1.0 / k)


def weibull_ppf(p: np.ndarray, lam: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Обратная функция распределения Вейбулла: U = λ·(-ln(1-p))^{1/k}."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return lam * (-np.log1p(-p)) ** (1.0 / k)


def matern32_corr(dh: np.ndarray, L: float) -> np.ndarray:
    """Корреляционная функция фильтра 2-го порядка: (1 + α|Δh|)·exp(-α|Δh|), α = 1/L."""
    x = np.abs(dh) / L
    return (1.0 + x) * np.exp(-x)


# ---------------------------------------------------------------------------
# Формирующий фильтр по высоте
# ---------------------------------------------------------------------------
class ShapingFilter:
    """
    Генератор реализаций гауссова процесса ξ(h) с M = 0, D = 1 на сетке h_grid [м],
    L_corr [м] может зависеть от высоты.

    Коэффициенты дискретного фильтра зависят только от сетки и L_corr, поэтому
    считаются один раз в конструкторе; sample() — только рекуррентный проход.

    order=1 (Орнштейн–Уленбек):
        ξ_{n+1} = ρ_n ξ_n + √(1-ρ_n²) ε_n,           ρ_n = exp(-α_{n+1} Δ)
    order=2 (кратный полюс, Матерн 3/2), состояние x = (ξ, ξ'):
        x_{n+1} = Φ_n x_n + L_n ε_n,   ε_n ~ N(0, I₂),
        Φ = e^{-αΔ} [[1+αΔ, Δ], [-α²Δ, 1-αΔ]],
        Q = P∞ - Φ P∞ Φᵀ,  P∞ = diag(1, α²),  L_n L_nᵀ = Q (Холецкий).
    Порядок потребления случайных чисел совпадает с прежней функцией shaping_filter.
    """

    def __init__(self, h_grid: np.ndarray, L_corr: np.ndarray, order: int = 2):
        if order not in (1, 2):
            raise ValueError("order должен быть 1 или 2")
        h_grid = np.asarray(h_grid, dtype=float)
        L_corr = np.broadcast_to(np.asarray(L_corr, dtype=float), h_grid.shape)
        self.order = order
        self.n = h_grid.size
        alpha = 1.0 / L_corr
        d = np.diff(h_grid)
        a = alpha[1:]
        self.alpha0 = float(alpha[0])

        if order == 1:
            rho = np.exp(-a * d)
            self._c = (rho.tolist(), np.sqrt(1.0 - rho * rho).tolist())
            return

        e = np.exp(-a * d)
        Phi = np.empty((self.n - 1, 2, 2))
        Phi[:, 0, 0] = e * (1.0 + a * d)
        Phi[:, 0, 1] = e * d
        Phi[:, 1, 0] = -e * a * a * d
        Phi[:, 1, 1] = e * (1.0 - a * d)
        P = np.zeros((self.n - 1, 2, 2))
        P[:, 0, 0] = 1.0
        P[:, 1, 1] = a * a
        Q = P - Phi @ P @ np.transpose(Phi, (0, 2, 1))
        Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1))) + 1e-15 * np.eye(2)
        Lq = np.linalg.cholesky(Q)
        # при смене α производная ξ' пересчитывается на новую стационарную дисперсию α²
        scale = np.ones(self.n - 1)
        ch = alpha[1:] != alpha[:-1]
        scale[ch] = alpha[1:][ch] / alpha[:-1][ch]
        self._c = tuple(v.tolist() for v in (Phi[:, 0, 0], Phi[:, 0, 1], Phi[:, 1, 0], Phi[:, 1, 1],
                                             Lq[:, 0, 0], Lq[:, 1, 0], Lq[:, 1, 1], scale))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        out = [0.0] * n
        if self.order == 1:
            z = rng.standard_normal(n).tolist()
            rho, sq = self._c
            x = z[0]
            out[0] = x
            for i in range(n - 1):
                x = rho[i] * x + sq[i] * z[i + 1]
                out[i + 1] = x
            return np.array(out)

        z = rng.standard_normal(2 * n).tolist()
        p00, p01, p10, p11, l00, l10, l11, sc = self._c
        x0, x1 = z[0], self.alpha0 * z[1]          # из стационарного распределения
        out[0] = x0
        for i in range(n - 1):
            x1 *= sc[i]
            z0, z1 = z[2 * i + 2], z[2 * i + 3]
            x0, x1 = (p00[i] * x0 + p01[i] * x1 + l00[i] * z0,
                      p10[i] * x0 + p11[i] * x1 + l10[i] * z0 + l11[i] * z1)
            out[i + 1] = x0
        return np.array(out)


def shaping_filter(h_grid: np.ndarray, L_corr: np.ndarray, rng: np.random.Generator,
                   order: int = 2) -> np.ndarray:
    """Совместимость с прежним интерфейсом (для многократных вызовов создайте ShapingFilter один раз)."""
    return ShapingFilter(h_grid, L_corr, order).sample(rng)


# ---------------------------------------------------------------------------
# Одна реализация поля ветра
# ---------------------------------------------------------------------------
@dataclass
class WindRealization:
    """Реализация ветра на сетке высот. Вызов wind(h) -> (u, v, w) в м/с, ENU."""
    h: np.ndarray            # сетка высот, м (по возрастанию, равномерная)
    u: np.ndarray            # восточная компонента (отрицательная = на запад)
    v: np.ndarray            # северная
    w: np.ndarray            # вертикальная (вверх)
    _dh: float = field(init=False, repr=False)
    _uL: list = field(init=False, repr=False)
    _vL: list = field(init=False, repr=False)
    _wL: list = field(init=False, repr=False)

    def __post_init__(self):
        self._dh = float(self.h[1] - self.h[0])
        # списки Python быстрее в скалярном цикле, чем индексирование numpy
        self._uL, self._vL, self._wL = self.u.tolist(), self.v.tolist(), self.w.tolist()

    def __call__(self, h_m: float):
        """Линейная интерполяция по высоте (непрерывна → корректна внутри шагов РК4)."""
        n = len(self._uL)
        s = h_m / self._dh
        if s <= 0.0:
            return self._uL[0], self._vL[0], self._wL[0]
        if s >= n - 1:
            return self._uL[-1], self._vL[-1], self._wL[-1]
        i = int(s)
        f = s - i
        u, v, w = self._uL, self._vL, self._wL
        return (u[i] + f * (u[i + 1] - u[i]),
                v[i] + f * (v[i + 1] - v[i]),
                w[i] + f * (w[i + 1] - w[i]))

    def mean_vector(self, h_low: float, h_high: float):
        """Средний вектор горизонтального ветра (u, v) в слое [h_low, h_high], м."""
        msk = (self.h >= h_low) & (self.h <= h_high)
        return float(self.u[msk].mean()), float(self.v[msk].mean())

    def in_trajectory_frame(self, h_m: float, azimuth_rad: float):
        """
        Проекции ветра на оси траекторной системы:
            Wx — горизонтально вдоль трассы (по направлению полёта),
            Wy — вверх,
            Wz — горизонтально поперёк трассы (вправо).
        azimuth_rad — азимут трассы от севера по часовой стрелке.
        """
        u, v, w = self(h_m)
        sa, ca = math.sin(azimuth_rad), math.cos(azimuth_rad)
        return u * sa + v * ca, w, u * ca - v * sa


@dataclass
class VenusWindModel:
    """
    Генератор реализаций ветра.

    filter_order : 1 или 2 — порядок формирующего фильтра.
    dh           : шаг сетки по высоте, м.
    h_max        : верхняя граница сетки, м.
    latitude_deg : широта точки входа (ослабляет суперротацию у полюсов,
                   задаёт среднее меридиональное течение).
    use_weibull  : True — зональная скорость по Вейбуллу; False — гауссова
                   с тем же средним и СКО (для сравнения).
    two_scale    : True — крупномасштабная + турбулентная части (см. п. 4 в шапке);
                   False — прежняя одномасштабная модель с L_corr = Ls.
    """
    filter_order: int = 2
    dh: float = 50.0
    h_max: float = 130_000.0
    latitude_deg: float = 0.0
    use_weibull: bool = True
    two_scale: bool = True
    profile: np.ndarray = field(default_factory=lambda: PROFILE.copy())

    def __post_init__(self):
        self.h = np.arange(0.0, self.h_max + self.dh, self.dh)
        hk = self.h / 1000.0
        p = self.profile
        col = {name: np.interp(hk, p[:, 0], p[:, i]) for i, name in enumerate(PROFILE_COLUMNS)}
        self.U_mean = col["U"] * latitude_factor(self.latitude_deg)
        self.k = col["k"]
        self.sig_v = col["sv"]
        self.sig_w = col["sw"]
        self.L_small = col["Ls"] * 1000.0
        self.L_large = col["Ll"] * 1000.0
        self.sig_turb = col["st"]
        self.V_mean = col["Vm"] * meridional_factor(self.latitude_deg)
        self.L_corr = self.L_small          # имя из прежней версии

        self.lam = weibull_scale(self.U_mean, self.k)
        g1 = gamma_fn(1.0 + 1.0 / self.k)
        g2 = gamma_fn(1.0 + 2.0 / self.k)
        self.U_std = self.lam * np.sqrt(np.maximum(g2 - g1 ** 2, 0.0))

        # доли дисперсии турбулентной части: β = (σ_турб / σ)², 0 ≤ β ≤ 1
        def frac(st, s):
            return np.clip(np.divide(st ** 2, s ** 2, out=np.ones_like(s), where=s > 0), 0.0, 1.0)
        self.beta_u = frac(self.sig_turb, self.U_std)
        self.beta_v = frac(self.sig_turb, self.sig_v)

        self._f_small = ShapingFilter(self.h, self.L_small, self.filter_order)
        self._f_large = ShapingFilter(self.h, self.L_large, self.filter_order) if self.two_scale else None

    # -- случайные процессы ξ ~ N(0,1) в каждом сечении ----------------------
    def _xi(self, rng, beta):
        if not self.two_scale:
            return self._f_small.sample(rng)
        xi_l = self._f_large.sample(rng)
        xi_s = self._f_small.sample(rng)
        return np.sqrt(1.0 - beta) * xi_l + np.sqrt(beta) * xi_s

    def sample(self, rng: np.random.Generator) -> WindRealization:
        xi_u = self._xi(rng, self.beta_u)
        xi_v = self._xi(rng, self.beta_v)
        xi_w = self._f_small.sample(rng)      # вертикальный ветер — только мелкомасштабный

        if self.use_weibull:
            U = weibull_ppf(ndtr(xi_u), self.lam, self.k)       # U = F_W^{-1}(Φ(ξ))
        else:
            U = np.maximum(self.U_mean + self.U_std * xi_u, 0.0)

        u = -U                                   # суперротация — на запад
        v = self.V_mean + self.sig_v * xi_v
        w = self.sig_w * xi_w
        return WindRealization(self.h, u, v, w)

    def mean_vector(self, h_low: float, h_high: float):
        """Климатический (математическое ожидание) вектор ветра (u, v) в слое, м/с."""
        msk = (self.h >= h_low) & (self.h <= h_high)
        return float(-self.U_mean[msk].mean()), float(self.V_mean[msk].mean())

    def corr_theory(self, j0: int, dh: np.ndarray, component: str = "u") -> np.ndarray:
        """Теоретическая корреляционная функция ξ на высоте h[j0] (параметры — локальные,
        формула точна в слое, где L и β постоянны; только для filter_order = 2)."""
        if not self.two_scale:
            return matern32_corr(dh, self.L_small[j0])
        b = (self.beta_u if component == "u" else self.beta_v)[j0]
        return (1 - b) * matern32_corr(dh, self.L_large[j0]) + b * matern32_corr(dh, self.L_small[j0])


def downwind_azimuth(u: float, v: float) -> float:
    """
    Азимут (от севера по часовой, рад, 0…2π) направления, КУДА дует ветер (u — восток, v — север).
    Для чисто западного ветра u < 0, v = 0 -> 270°.
    """
    return math.atan2(u, v) % (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Самопроверка: python venus_wind.py
# ---------------------------------------------------------------------------
def _self_check(n_real: int = 1000, seed: int = 2026, out_prefix: str = "wind_check"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    model = VenusWindModel()
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_real)]
    real = [model.sample(r) for r in rngs]
    U = np.array([-r.u for r in real])
    hk = model.h / 1000.0

    # 1) реализации + среднее и ±σ
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for r in real[:8]:
        ax[0].plot(-r.u, hk, lw=0.8)
        ax[1].plot(r.v, hk, lw=0.8)
        ax[2].plot(r.w, hk, lw=0.8)
    ax[0].plot(model.U_mean, hk, "k", lw=2, label="Ū(h) — задано")
    ax[0].plot(U.mean(0), hk, "k--", lw=1.5, label="среднее по реализациям")
    ax[0].fill_betweenx(hk, model.U_mean - model.U_std, model.U_mean + model.U_std,
                        color="0.8", label="±σ (Вейбулл)")
    ax[0].set_xlabel("Зональная скорость (на запад), м/с")
    ax[1].set_xlabel("Меридиональная v, м/с")
    ax[2].set_xlabel("Вертикальная w, м/с")
    ax[0].set_ylabel("Высота, км")
    ax[0].legend(loc="lower right", fontsize=8)
    for a in ax:
        a.grid(True)
    fig.suptitle("Реализации ветра (двухмасштабный формирующий фильтр + закон Вейбулла)")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_profiles.png", dpi=130)
    plt.close(fig)

    # 2) гистограммы на нескольких высотах vs плотность Вейбулла
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    report = []
    for a, h0 in zip(ax, (10.0, 50.0, 68.0)):
        j = int(round(h0 * 1000 / model.dh))
        sample = U[:, j]
        a.hist(sample, bins=30, density=True, alpha=0.6, label="моделирование")
        xs = np.linspace(0, sample.max() * 1.1, 300)
        a.plot(xs, stats.weibull_min.pdf(xs, model.k[j], scale=model.lam[j]), "r",
               label=f"Вейбулл k={model.k[j]:.1f}, λ={model.lam[j]:.1f}")
        ks = stats.kstest(sample, "weibull_min", args=(model.k[j], 0, model.lam[j]))
        a.set_title(f"h = {h0:.0f} км, KS p = {ks.pvalue:.2f}")
        a.set_xlabel("U, м/с")
        a.legend(fontsize=8)
        a.grid(True)
        report.append((h0, sample.mean(), model.U_mean[j], sample.std(), model.U_std[j], ks.pvalue))
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_weibull.png", dpi=130)
    plt.close(fig)

    # 3) корреляционная функция (слой 26–40 км: L и β постоянны)
    j0 = int(26_000 / model.dh)
    lags = np.arange(0, int(14_000 / model.dh))
    xs_ = (U - U.mean(0)) / U.std(0)
    emp = np.array([np.mean(xs_[:, j0] * xs_[:, j0 + l]) for l in lags])
    dh_ = lags * model.dh
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dh_ / 1000, emp, "o", ms=3, label="оценка по реализациям")
    ax.plot(dh_ / 1000, model.corr_theory(j0, dh_), "r", label="(1-β)K_L + βK_S")
    ax.plot(dh_ / 1000, matern32_corr(dh_, model.L_small[j0]), "r:", label=f"K_S, L = {model.L_small[j0] / 1e3:.0f} км")
    ax.set_xlabel("Δh, км")
    ax.set_ylabel("Нормированная корреляция U")
    ax.set_title(f"Корреляционная функция зонального ветра, h = 26 км, β = {model.beta_u[j0]:.3f}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_corr.png", dpi=130)
    plt.close(fig)

    print("h, км | M[U] модель / задано | σ[U] модель / теор. | p-value KS")
    for h0, m_, m0, s_, s0, p in report:
        print(f"{h0:5.0f} | {m_:7.2f} / {m0:7.2f} | {s_:6.2f} / {s0:6.2f} | {p:.3f}")
    print(f"max|K_эмп - K_теор| (26 км) = {np.abs(emp - model.corr_theory(j0, dh_)).max():.3f}")


if __name__ == "__main__":
    _self_check()
