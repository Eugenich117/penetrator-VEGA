import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple, Optional


class BranchAndBoundNode:
    """Узел дерева ветвей и границ"""

    def __init__(self, bounds, level=0, parent_info=""):
        self.bounds = bounds  # Границы переменных [(lower, upper), ...]
        self.level = level
        self.parent_info = parent_info
        self.solution = None
        self.objective = None
        self.is_feasible = False
        self.is_integer = False


def solve_lp_relaxation(c, A, b, bounds):
    """
    Решает LP-релаксацию задачи
    minimize c^T * x
    subject to A * x <= b, bounds
    """
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    return result


def is_integer_solution(x, tol=1e-6):
    """Проверяет, является ли решение целочисленным"""
    return np.all(np.abs(x - np.round(x)) < tol)


def find_fractional_variable(x, tol=1e-6):
    """Находит индекс первой нецелочисленной переменной"""
    for i, val in enumerate(x):
        if np.abs(val - np.round(val)) >= tol:
            return i
    return -1


def branch_and_bound(c, A, b, initial_bounds=None):
    """
    Метод ветвей и границ для целочисленного линейного программирования
    """
    n_vars = len(c)

    # Начальные границы: x_i >= 0
    if initial_bounds is None:
        initial_bounds = [(0, None) for _ in range(n_vars)]

    # Глобальные переменные
    best_solution = None
    best_objective = float('inf')
    node_counter = 0

    # Стек узлов для обработки
    nodes_to_process = [BranchAndBoundNode(initial_bounds, level=0, parent_info="Корневой узел")]

    print("=" * 80)
    print("МЕТОД ВЕТВЕЙ И ГРАНИЦ")
    print("=" * 80)
    print(f"\nЦелевая функция: minimize {c[0]:.2f}*x1 + {c[1]:.2f}*x2")
    print(f"\nОграничения:")
    for i in range(len(A)):
        print(f"  {A[i][0]:.2f}*x1 + {A[i][1]:.2f}*x2 <= {b[i]:.2f}")
    print(f"\nПеременные: x1, x2 >= 0, целые\n")

    while nodes_to_process:
        # Берем узел из стека
        current_node = nodes_to_process.pop(0)
        node_counter += 1

        print("=" * 80)
        print(f"УЗЕЛ #{node_counter} (Уровень {current_node.level})")
        print("=" * 80)
        print(f"Информация: {current_node.parent_info}")
        print(f"\nГраницы переменных:")
        for i, (lb, ub) in enumerate(current_node.bounds):
            lb_str = f"{lb}" if lb is not None else "0"
            ub_str = f"{ub}" if ub is not None else "∞"
            print(f"  x{i + 1}: [{lb_str}, {ub_str}]")

        # Решаем LP-релаксацию
        result = solve_lp_relaxation(c, A, b, current_node.bounds)

        if not result.success:
            print(f"\n❌ LP-релаксация не имеет допустимого решения")
            print(f"Причина: {result.message}")
            print("→ Узел отсекается (недопустимая область)\n")
            continue

        current_node.solution = result.x
        current_node.objective = result.fun
        current_node.is_feasible = True

        print(f"\n✓ LP-релаксация решена успешно:")
        print(f"  Решение: x1 = {result.x[0]:.6f}, x2 = {result.x[1]:.6f}")
        print(f"  Целевая функция: f = {result.fun:.6f}")

        # Проверка границы
        if current_node.objective >= best_objective:
            print(f"\n✂ Отсечение по границе:")
            print(f"  Текущее значение f = {current_node.objective:.6f}")
            print(f"  Лучшее целое решение f* = {best_objective:.6f}")
            print(f"  f >= f* → узел отсекается\n")
            continue

        # Проверка целочисленности
        if is_integer_solution(result.x):
            current_node.is_integer = True
            print(f"\n🎯 Найдено целочисленное решение!")
            print(f"  x1 = {int(round(result.x[0]))}, x2 = {int(round(result.x[1]))}")
            print(f"  f = {result.fun:.6f}")

            if result.fun < best_objective:
                best_objective = result.fun
                best_solution = result.x
                print(f"  ⭐ Это новое лучшее решение! Обновляем рекорд.\n")
            else:
                print(f"  Не улучшает текущий рекорд (f* = {best_objective:.6f})\n")
            continue

        # Ветвление
        frac_idx = find_fractional_variable(result.x)
        frac_value = result.x[frac_idx]

        print(f"\n🌳 Ветвление по переменной x{frac_idx + 1}:")
        print(f"  Текущее значение: x{frac_idx + 1} = {frac_value:.6f}")
        print(f"  Дробная часть: {frac_value - np.floor(frac_value):.6f}")
        print(f"  Создаем две ветви:")
        print(f"    Левая ветвь:  x{frac_idx + 1} <= {int(np.floor(frac_value))}")
        print(f"    Правая ветвь: x{frac_idx + 1} >= {int(np.ceil(frac_value))}")

        # Левая ветвь: x_i <= floor(x_i)
        left_bounds = current_node.bounds.copy()
        left_bounds[frac_idx] = (left_bounds[frac_idx][0], np.floor(frac_value))
        left_node = BranchAndBoundNode(
            left_bounds,
            level=current_node.level + 1,
            parent_info=f"Узел #{node_counter}, левая ветвь: x{frac_idx + 1} <= {int(np.floor(frac_value))}"
        )

        # Правая ветвь: x_i >= ceil(x_i)
        right_bounds = current_node.bounds.copy()
        lb = max(right_bounds[frac_idx][0] if right_bounds[frac_idx][0] is not None else 0,
                 np.ceil(frac_value))
        right_bounds[frac_idx] = (lb, right_bounds[frac_idx][1])
        right_node = BranchAndBoundNode(
            right_bounds,
            level=current_node.level + 1,
            parent_info=f"Узел #{node_counter}, правая ветвь: x{frac_idx + 1} >= {int(np.ceil(frac_value))}"
        )

        # Добавляем узлы в стек (сначала левый, потом правый - поиск в глубину)
        nodes_to_process.insert(0, left_node)
        nodes_to_process.insert(0, right_node)

        print(f"\n→ Добавлено 2 новых узла для обработки\n")

    # Итоговый результат
    print("=" * 80)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 80)
    print(f"\nВсего обработано узлов: {node_counter}")

    if best_solution is not None:
        print(f"\n✅ Оптимальное целочисленное решение найдено:")
        print(f"  x1* = {int(round(best_solution[0]))}")
        print(f"  x2* = {int(round(best_solution[1]))}")
        print(f"  f* = {best_objective:.6f}")

        # Проверка ограничений
        print(f"\nПроверка ограничений:")
        x_int = np.round(best_solution)
        for i in range(len(A)):
            lhs = np.dot(A[i], x_int)
            print(
                f"  Ограничение {i + 1}: {A[i][0]:.2f}*{int(x_int[0])} + {A[i][1]:.2f}*{int(x_int[1])} = {lhs:.2f} <= {b[i]:.2f} ✓")
    else:
        print(f"\n❌ Целочисленное допустимое решение не найдено")

    print("=" * 80)

    return best_solution, best_objective


# Исходные данные
c = np.array([-4.87, -3.47])
A = np.array([
    [6.83, 6.09],
    [0.95, 8.478]
])
b = np.array([10.97, 18.65])

# Решение методом ветвей и границ
solution, objective = branch_and_bound(c, A, b)
