import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Определение функции и её градиента
class QuadraticFunction:
    def __init__(self):
        # Матрица H (2x2, симметричная)
        self.H = np.array([[6, -5],
                           [-5, 10]])

        # Вектор c (2x1)
        self.c = np.array([10, -50])

        # Матрица ограничений G (3x2)
        self.G = np.array([[1, 1],
                           [-1, 2],
                           [0, -1]])

        # Вектор правой части b (3x1)
        self.b = np.array([5, 6, 2])

        # Коэффициенты штрафа P (3x1)
        self.P = np.array([0, 0, 0])  # Изначально нулевые, будут обновляться

        # Глобальный минимум (аналитическое решение)
        # Решаем x = 0.5 * H^(-1) * C, как в btn2Click
        self.H_inv = np.linalg.inv(self.H)
        self.global_min = 0.5 * np.dot(self.H_inv, self.c)

    def evaluate(self, x):
        # Основная функция: x^T H x + c^T x
        f = np.dot(x.T, np.dot(self.H, x)) + np.dot(self.c.T, x)

        # Штрафы за нарушение ограничений Gx <= b
        penalty = 0
        constraints = np.dot(self.G, x) - self.b
        for i in range(len(constraints)):
            if constraints[i] > 0:
                penalty += self.P[i] * constraints[i] ** 2
        return f + penalty

    def gradient(self, x):
        # Градиент функции: Hx + c + градиент штрафа
        grad = np.dot(self.H, x) + self.c
        constraints = np.dot(self.G, x) - self.b
        for i in range(len(constraints)):
            if constraints[i] > 0:
                grad += 2 * self.P[i] * constraints[i] * self.G[i]
        return grad

    def evaluate_without_penalty(self, x):
        # Функция без штрафа для аналитического минимума
        return np.dot(x.T, np.dot(self.H, x)) + np.dot(self.c.T, x)

# Функция для численной оптимизации
def objective(x, func):
    return func.evaluate(x)

# Аналитическое решение для стационарных точек
def analytical_solution(func):
    results = {}

    # Безусловный минимум
    x0 = func.global_min
    y0 = func.evaluate_without_penalty(x0)
    results['Безусловный минимум'] = (x0, y0)

    # Угловые точки
    G_subsets = [
        func.G[1:3, :],  # G1 и G3
        func.G[[0, 2], :],  # G0 и G3
        func.G[0:2, :]   # G0 и G1
    ]
    b_subsets = [
        func.b[1:3],
        func.b[[0, 2]],
        func.b[0:2]
    ]

    for i, (G_sub, b_sub) in enumerate(zip(G_subsets, b_subsets), 1):
        try:
            AP = np.linalg.solve(G_sub, b_sub)
            y_AP = func.evaluate_without_penalty(AP)
            results[f'Угловая точка {i}'] = (AP, y_AP)
        except np.linalg.LinAlgError:
            continue

    # Стационарные точки с лямбда
    A_matrices = [
        np.array([[12, 25, 1],
                  [25, 20, 1],
                  [1, 1, 0]]),
        np.array([[12, 25, 0],
                  [25, 20, -1],
                  [0, -1, 0]]),
        np.array([[12, 25, 1],
                  [25, 20, -2],
                  [1, -2, 0]])
    ]
    E_vectors = [
        np.array([-10, 50, 5]),
        np.array([-10, 50, 0]),
        np.array([-10, 50, -6])
    ]

    for i, (A, E) in enumerate(zip(A_matrices, E_vectors), 1):
        try:
            SP = np.linalg.solve(A, E)
            x_SP = SP[:2]
            y_SP = func.evaluate_without_penalty(x_SP)
            results[f'Стационарная точка {i}'] = (SP, y_SP)
        except np.linalg.LinAlgError:
            continue

    return results

# Основная программа
# Основная программа
def main():
    func = QuadraticFunction()

    # Начальные значения , по умолчанию [-10, 10]
    x0 = np.array([-10.0, 10.0])

    # Коэффициенты штрафа, по умолчанию [0, 0, 0]
    P = np.array([0.0, 0.0, 0.0])
    func.P = P

    # Численные методы оптимизации
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    results = {}

    for method in methods:
        # Используем lambda для корректной передачи func в градиент
        if method == 'COBYLA':
            res = minimize(objective, x0, args=(func,), method=method, jac=lambda x, f=func: f.gradient(x))
        else:
            res = minimize(objective, x0, args=(func,), method=method)
        results[method] = (res.x, res.fun)

    # Аналитическое решение
    analytical_results = analytical_solution(func)

    # Вывод результатов
    print("Численные методы оптимизации:")
    for method, (x, y) in results.items():
        print(f"{method}: x = [{x[0]:.3f}, {x[1]:.3f}], f(x) = {y:.3f}")

    print("\nАналитические решения:")
    for name, (x, y) in analytical_results.items():
        if x.shape[0] == 2:
            print(f"{name}: x = [{x[0]:.3f}, {x[1]:.3f}], f(x) = {y:.3f}")
        else:
            print(f"{name}: x = [{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}], f(x) = {y:.3f}")

    # Исправленная часть: Построение графика
    # Собираем все координаты x1 и x2
    all_x1 = [float(x[0]) for x, _ in results.values()] + [float(r[0][0]) for r in analytical_results.values() if r[0].shape[0] == 2]
    all_x2 = [float(x[1]) for x, _ in results.values()] + [float(r[0][1]) for r in analytical_results.values() if r[0].shape[0] == 2]

    # Отладочный вывод для проверки содержимого списков
    '''    print("all_x1:", all_x1)
    print("all_x2:", all_x2)
    print("Length of all_x1:", len(all_x1))
    print("Length of all_x2:", len(all_x2))'''

    # Определяем границы с небольшим отступом (20% от диапазона)
    if not all_x1 or not all_x2:
        raise ValueError("all_x1 or all_x2 is empty. Check results and analytical_results.")
    x1_min, x1_max = min(all_x1), max(all_x1)
    x2_min, x2_max = min(all_x2), max(all_x2)

    x1_range = x1_max - x1_min
    x2_range = x2_max - x2_min
    print(f"x1_range: {x1_range}, x2_range: {x2_range}")

    # Добавляем отступ
    padding = 0.5  # 20% от диапазона
    x1_min -= x1_range * padding
    x1_max += x1_range * padding
    x2_min -= x2_range * padding
    x2_max += x2_range * padding

    # Проверяем, чтобы диапазон не был слишком маленьким
    if x1_range < 1e-6:
        x1_min, x1_max = x1_min - 1, x1_max + 1
    if x2_range < 1e-6:
        x2_min, x2_max = x2_min - 1, x2_max + 1

    #print(f"Final x1 range: [{x1_min}, {x1_max}], x2 range: [{x2_min}, {x2_max}]")

    # Создаём сетку на основе новых границ
    x1 = np.linspace(x1_min, x1_max, 200)  # Увеличиваем количество точек для более плавных линий
    x2 = np.linspace(x2_min, x2_max, 200)
    X1, X2 = np.meshgrid(x1, x2)

    # Вычисляем Z с отладочным выводом
    Z = np.array([[func.evaluate_without_penalty(np.array([x1_val, x2_val])) for x1_val, x2_val in zip(row_x1, row_x2)]
                  for row_x1, row_x2 in zip(X1, X2)])
    #print("Z shape:", Z.shape)
    #print("Z min:", np.min(Z), "Z max:", np.max(Z))

    # Настраиваем уровни контурных линий
    z_min, z_max = np.min(Z), np.max(Z)
    levels = np.linspace(z_min, z_max, 20)  # Подбираем уровни динамически

    # Создаём график
    plt.figure(figsize=(10, 6))
    contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)  # Добавляем подписи к контурным линиям

    # Визуализация области допустимых решений (Gx <= b)
    # Ограничение 1: x1 + x2 <= 5
    x1_constraint1 = np.linspace(x1_min, x1_max, 100)
    x2_constraint1 = 5 - x1_constraint1
    plt.plot(x1_constraint1, x2_constraint1, 'b--', label='x1 + x2 = 5')

    # Ограничение 2: -x1 + 2x2 <= 6
    x2_constraint2 = (6 + x1_constraint1) / 2
    plt.plot(x1_constraint1, x2_constraint2, 'm--', label='-x1 + 2x2 = 6')

    # Ограничение 3: -x2 <= 2 (или x2 >= -2)
    plt.axhline(y=-2, color='c', linestyle='--', label='x2 = -2')

    # Корректное извлечение координат для численных минимумов
    num_x = [float(x[0]) for x, _ in results.values()]
    num_y = [float(x[1]) for x, _ in results.values()]
    plt.scatter(num_x, num_y, c='r', label='Численные минимумы', zorder=5)

    # Корректное извлечение координат для аналитических точек
    anal_x = [float(r[0][0]) for r in analytical_results.values() if r[0].shape[0] == 2]
    anal_y = [float(r[0][1]) for r in analytical_results.values() if r[0].shape[0] == 2]
    plt.scatter(anal_x, anal_y, c='g', label='Угловые точки', zorder=5)

    # Добавляем оси координат через (0;0)
    ax = plt.gca()  # Получаем текущие оси
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Горизонтальная ось
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)  # Вертикальная ось

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Контурная карта функции и минимумы')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

