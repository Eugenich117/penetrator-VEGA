import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import time
from mpl_toolkits.mplot3d import Axes3D


def linear_programming_simplex():
    """Основная функция для решения задачи линейного программирования симплекс-методом"""

    # Вызов симплекс-метода
    results = solve_with_simplex()

    # Визуализация результатов
    visualize_solution(results)


def solve_with_simplex():
    """Решение задачи симплекс-методом с использованием scipy.optimize.linprog"""

    print('=== СИМПЛЕКС-МЕТОД ===')

    # Параметры задачи в стандартной форме:
    # Минимизация: min f'*x
    # Наша задача: max (-4.87*x1 - 3.47*x2) = min (4.87*x1 + 3.47*x2)

    # Коэффициенты целевой функции (для минимизации)
    f = [4.87, 3.47]

    # Матрица ограничений неравенств A*x <= b
    A = [[6.83, 6.09],  # 6.83*x1 + 6.09*x2 <= 10.97
         [0.95, 8.478]]  # 0.95*x1 + 8.478*x2 <= 18.65
    b = [10.97, 18.65]

    # Границы переменных
    bounds = [(0, None), (0, None)]  # x1 >= 0, x2 >= 0

    # Решение задачи
    start_time = time.time()
    result = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method='simplex')
    computation_time = time.time() - start_time

    # Формирование результатов
    results = {
        'x_opt': result.x,
        'f_opt_original': result.fun,
        'success': result.success,
        'status': result.status,
        'nit': result.nit,
        'computation_time': computation_time
    }

    # Вывод результатов
    print('\n--- РЕЗУЛЬТАТЫ ---')

    if result.success:
        print('Решение найдено успешно!')
        print('Оптимальные значения:')
        print(f'  x1 = {result.x[0]:.6f}')
        print(f'  x2 = {result.x[1]:.6f}')
        print(f'Значение целевой функции: {result.fun:.6f}')
        print(f'Количество итераций: {result.nit}')
        print(f'Время вычисления: {computation_time:.4f} секунд')

        # Проверка ограничений
        print('\n--- ПРОВЕРКА ОГРАНИЧЕНИЙ ---')
        constraint1_value = A[0][0] * result.x[0] + A[0][1] * result.x[1]
        constraint2_value = A[1][0] * result.x[0] + A[1][1] * result.x[1]
        print(
            f'Ограничение 1: 6.83*x1 + 6.09*x2 = {constraint1_value:.6f} <= 10.97 ({check_constraint(constraint1_value, 10.97)})')
        print(
            f'Ограничение 2: 0.95*x1 + 8.478*x2 = {constraint2_value:.6f} <= 18.65 ({check_constraint(constraint2_value, 18.65)})')
        print(
            f'Неотрицательность: x1 = {result.x[0]:.6f} >= 0, x2 = {result.x[1]:.6f} >= 0 ({check_nonnegativity(result.x)})')

    else:
        print(f'Решение не найдено. Статус: {result.status}')
        if result.status == 1:
            print('Превышено максимальное количество итераций.')
        elif result.status == 2:
            print('Задача несовместна (нет допустимых решений).')
        elif result.status == 3:
            print('Задача неограничена.')
        elif result.status == 4:
            print('Задача неопределена.')
        else:
            print('Неизвестная ошибка.')

    print()
    return results


def check_constraint(value, limit):
    """Проверка выполнения ограничения"""
    if value <= limit + 1e-6:
        return 'выполнено'
    else:
        return 'НЕ ВЫПОЛНЕНО'


def check_nonnegativity(x):
    """Проверка неотрицательности переменных"""
    if all(x_i >= -1e-6 for x_i in x):
        return 'выполнено'
    else:
        return 'НЕ ВЫПОЛНЕНО'


def visualize_solution(results):
    """Визуализация области допустимых решений и оптимальной точки"""

    if not results['success']:
        print('Визуализация невозможна - решение не найдено.')
        return

    print('=== ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ ===')

    # Находим точку пересечения ограничений
    A_intersect = np.array([[6.83, 6.09], [0.95, 8.478]])
    b_intersect = np.array([10.97, 18.65])
    X_intersect = np.linalg.solve(A_intersect, b_intersect)

    print('Точка пересечения ограничений:')
    print(f'x1 = {X_intersect[0]:.6f}, x2 = {X_intersect[1]:.6f}')

    # Создание сетки для визуализации
    x1 = np.linspace(-0.5, 3, 200)
    x2 = np.linspace(0, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)

    # Целевая функция
    Z = -4.87 * X1 - 3.47 * X2

    # Область допустимых решений
    feasible = (6.83 * X1 + 6.09 * X2 <= 10.97) & \
               (0.95 * X1 + 8.478 * X2 <= 18.65) & \
               (X1 >= 0) & (X2 >= 0)

    # Создание графиков
    fig = plt.figure(figsize=(15, 12))

    # 1. Область допустимых решений с улучшенной визуализацией
    ax1 = plt.subplot(2, 2, 1)

    # Построение линий ограничений
    x1_plot = np.linspace(-0.5, 2.5, 200)
    x2_1 = (10.97 - 6.83 * x1_plot) / 6.09
    x2_2 = (18.65 - 0.95 * x1_plot) / 8.478

    plt.plot(x1_plot, x2_1, color='red', linewidth=3, label='6.83x1 + 6.09x2 = 10.97')
    plt.plot(x1_plot, x2_2, color='blue', linewidth=3, label='0.95x1 + 8.478x2 = 18.65')

    # Заливка ОДР
    x2_max = np.minimum(x2_1, x2_2)
    valid_indices = (x2_max >= 0) & (x1_plot >= 0)
    fill_x = np.concatenate([x1_plot[valid_indices], x1_plot[valid_indices][::-1]])
    fill_y = np.concatenate([np.zeros(np.sum(valid_indices)), x2_max[valid_indices][::-1]])
    plt.fill(fill_x, fill_y, 'yellow', alpha=0.3, label='ОДР')

    # Отметка оптимальной точки
    plt.plot(results['x_opt'][0], results['x_opt'][1], 'ko', markersize=10,
             markerfacecolor='black', label='X* (непрерыв.)')

    # Отметка точки пересечения ограничений
    plt.plot(X_intersect[0], X_intersect[1], 'gs', markersize=10,
             markerfacecolor='green', label='Пересечение ограничений')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Область допустимых решений ЗЛП')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([-0.5, 2.5])
    plt.ylim([0, 2.5])

    # 2. 3D визуализация целевой функции
    ax2 = plt.subplot(2, 2, 2, projection='3d')

    surf = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7,
                            linewidth=0, antialiased=True)
    ax2.scatter(results['x_opt'][0], results['x_opt'][1], results['f_opt_original'],
                color='red', s=100, marker='o')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('Целевая функция')
    ax2.set_title('3D визуализация целевой функции')

    # 3. Контурный график с целевой функцией
    ax3 = plt.subplot(2, 2, 3)

    contour = plt.contourf(X1, X2, Z, 50, cmap='viridis')

    # Границы ограничений
    plt.contour(X1, X2, 6.83 * X1 + 6.09 * X2, levels=[10.97], colors='red', linewidths=2)
    plt.contour(X1, X2, 0.95 * X1 + 8.478 * X2, levels=[18.65], colors='blue', linewidths=2)

    # Оптимальная точка
    plt.plot(results['x_opt'][0], results['x_opt'][1], 'ro', markersize=10,
             markerfacecolor='red')

    plt.colorbar(contour)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Контурный график целевой функции')
    plt.grid(True)

    # 4. Информация о решении
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    constraint1_value = 6.83 * results['x_opt'][0] + 6.09 * results['x_opt'][1]
    constraint2_value = 0.95 * results['x_opt'][0] + 8.478 * results['x_opt'][1]

    info_text = (f'ОПТИМАЛЬНОЕ РЕШЕНИЕ:\n\n'
                 f'x1 = {results["x_opt"][0]:.6f}\n'
                 f'x2 = {results["x_opt"][1]:.6f}\n\n'
                 f'Целевая функция: {results["f_opt_original"]:.6f}\n'
                 f'Итераций: {results["nit"]}\n'
                 f'Время: {results["computation_time"]:.4f} с\n\n'
                 f'Точка пересечения:\n'
                 f'x1 = {X_intersect[0]:.6f}\nx2 = {X_intersect[1]:.6f}\n\n'
                 f'Проверка ограничений:\n'
                 f'6.83*x1 + 6.09*x2 = {constraint1_value:.6f} <= 10.97\n'
                 f'0.95*x1 + 8.478*x2 = {constraint2_value:.6f} <= 18.65')

    ax4.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray",
                       edgecolor="black", alpha=0.8))
    ax4.set_title('Информация о решении')

    plt.tight_layout()
    plt.show()

def get_feasible_region_vertices():
    """Нахождение вершин допустимой области"""

    # Начало координат
    v1 = [0, 0]

    # Пересечение с осью x2
    v2 = [0, min(10.97 / 6.09, 18.65 / 8.478)]

    # Пересечение ограничений (если в неотрицательной области)
    A_eq = np.array([[6.83, 6.09], [0.95, 8.478]])
    b_eq = np.array([10.97, 18.65])
    sol = np.linalg.solve(A_eq, b_eq)
    if sol[0] >= 0 and sol[1] >= 0:
        v3 = sol
    else:
        v3 = None

    # Пересечение с осью x1
    v4 = [min(10.97 / 6.83, 18.65 / 0.95), 0]

    # Собираем все допустимые вершины
    vertices = [v1, v2]
    if v3 is not None:
        vertices.append(v3)
    vertices.append(v4)

    vertices = np.array(vertices)

    # Сортируем по углу для правильной заливки
    if len(vertices) > 2:
        center = np.mean(vertices, axis=0)
        angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
        order = np.argsort(angles)
        vertices = vertices[order, :]

    return vertices


# Запуск основной функции
if __name__ == "__main__":
    linear_programming_simplex()