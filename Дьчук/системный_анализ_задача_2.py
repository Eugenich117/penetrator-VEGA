import numpy as np
from scipy.optimize import linprog
import numpy as np


def simplex_method(f, A, b):
    """
    Реализация симплекс-метода для задачи минимизации
    min f'*x при условиях A*x <= b, x >= 0
    """
    # Добавляем slack переменные
    m, n = len(A), len(f)
    tableau = np.zeros((m + 1, n + m + 1))

    # Заполняем ограничения
    for i in range(m):
        tableau[i, :n] = A[i]
        tableau[i, n + i] = 1  # slack переменные
        tableau[i, -1] = b[i]

    # Заполняем целевую функцию (для минимизации)
    tableau[-1, :n] = f

    print("Начальная симплекс-таблица:")
    print(tableau)
    print()

    # Проверяем оптимальность (для минимизации - все коэффициенты в целевой строке >= 0)
    while np.any(tableau[-1, :-1] < 0):
        # Выбираем входящую переменную (наиболее отрицательный коэффициент)
        entering = np.argmin(tableau[-1, :-1])

        # Выбираем исходящую переменную (минимальное положительное отношение)
        ratios = []
        for i in range(m):
            if tableau[i, entering] > 0:
                ratio = tableau[i, -1] / tableau[i, entering]
                ratios.append(ratio)
            else:
                ratios.append(np.inf)

        if all(r == np.inf for r in ratios):
            print("Задача неограниченна!")
            return None

        leaving = np.argmin(ratios)

        print(f"Входящая переменная: x{entering + 1}")
        print(f"Исходящая переменная: s{leaving + 1}")
        print(f"Разрешающий элемент: {tableau[leaving, entering]}")

        # Нормализуем разрешающую строку
        pivot = tableau[leaving, entering]
        tableau[leaving, :] /= pivot

        # Обновляем остальные строки
        for i in range(m + 1):
            if i != leaving:
                factor = tableau[i, entering]
                tableau[i, :] -= factor * tableau[leaving, :]

        print("Обновленная таблица:")
        print(tableau)
        print()

    # Извлекаем решение
    solution = np.zeros(n)
    for j in range(n):
        col = tableau[:-1, j]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == m - 1:
            row_idx = np.where(col == 1)[0][0]
            solution[j] = tableau[row_idx, -1]

    optimal_value = tableau[-1, -1]

    return solution, optimal_value


'''f = [4.87, 3.47]  # Целевая функция для минимизации
A = [[6.83, 6.09],  # Матрица ограничений
     [0.95, 8.478]]
b = [10.97, 18.65]  # Правые части ограничений'''
f = [-1.83, -2.38]
A = [[3.67, -2.94], [2.36, 3.18]]
b = [9.97, 15.5]
# Решаем задачу
solution, optimal_value = simplex_method(f, A, b)

print("РЕЗУЛЬТАТЫ:")
print(f"Оптимальное решение: x1 = {solution[0]:.4f}, x2 = {solution[1]:.4f}")
print(f"Минимальное значение целевой функции: {optimal_value:.4f}")

# Проверка ограничений
print("\nПРОВЕРКА ОГРАНИЧЕНИЙ:")
print(f"6.83*x1 + 6.09*x2 = {6.83 * solution[0] + 6.09 * solution[1]:.4f} <= 10.97")
print(f"0.95*x1 + 8.478*x2 = {0.95 * solution[0] + 8.478 * solution[1]:.4f} <= 18.65")
print(f"x1 = {solution[0]:.4f} >= 0, x2 = {solution[1]:.4f} >= 0")


# Решаем задачу линейного программирования
result = linprog(f, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])

print("РЕЗУЛЬТАТЫ (scipy.optimize.linprog):")
print(f"Статус: {result.message}")
print(f"Оптимальное решение: x1 = {result.x[0]:.4f}, x2 = {result.x[1]:.4f}")
print(f"Минимальное значение целевой функции: {result.fun:.4f}")