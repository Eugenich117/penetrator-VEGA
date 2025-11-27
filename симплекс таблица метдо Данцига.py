import numpy as np
from scipy.optimize import linprog


class SimplexSolver:
    def __init__(self, c, A, b):
        """
        Инициализация симплекс-решателя для МИНИМИЗАЦИИ
        c - коэффициенты целевой функции
        A - матрица ограничений
        b - правые части ограничений
        """
        # Для минимизации меняем знак целевой функции
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = self.A.shape
        self.two_phase = False
        self.artificial_vars = []

    def solve(self):
        """Основной метод решения с пошаговым выводом"""
        print("=" * 80)
        print("МЕТОД ДАНЦИГА (СИМПЛЕКС-МЕТОД) - ПОИСК МИНИМУМА")
        print("=" * 80)

        # Проверка начального допустимого решения
        if not self._check_initial_feasibility():
            print("Начальное решение недопустимо. Запуск двухфазного метода...")
            return self._two_phase_solve()
        else:
            print("Начальное решение допустимо. Запуск прямой оптимизации...")
            return self._phase2_solve()

    def _check_initial_feasibility(self):
        """Проверка начального допустимого решения"""
        print("\nПРОВЕРКА НАЧАЛЬНОГО ДОПУСТИМОГО РЕШЕНИЯ:")
        print(f"Правые части ограничений: {self.b}")

        if np.all(self.b >= -1e-10):
            print("✓ Все b_i ≥ 0 - начальное решение допустимо")
            return True
        else:
            negative_count = np.sum(self.b < -1e-10)
            print(f"✗ Обнаружено {negative_count} отрицательных b_i - начальное решение недопустимо")
            return False

    def _two_phase_solve(self):
        """Двухфазный симплекс-метод"""
        print("\n" + "=" * 80)
        print("ФАЗА 1: Поиск начального допустимого решения")
        print("=" * 80)

        # Создаем задачу для фазы 1
        c_phase1 = self._create_phase1_objective()
        tableau_phase1 = self._create_phase1_tableau(c_phase1)

        print("Целевая функция фазы 1: min W = сумма искусственных переменных")
        print("Искусственные переменные:", self.artificial_vars)

        # Решаем фазу 1
        solution_phase1, w_value = self._solve_phase(tableau_phase1, "Фаза 1")

        if w_value > 1e-10:
            print("\n✗ Фаза 1 не удалась: W > 0, допустимого решения не существует")
            return None, None

        print("\n✓ Фаза 1 завершена: W = 0, начальное допустимое решение найдено")

        # Переход к фазе 2
        print("\n" + "=" * 80)
        print("ФАЗА 2: Оптимизация исходной целевой функции")
        print("=" * 80)

        tableau_phase2 = self._create_phase2_tableau(tableau_phase1)
        return self._solve_phase(tableau_phase2, "Фаза 2")

    def _create_phase1_objective(self):
        """Создание целевой функции для фазы 1"""
        # Целевая функция фазы 1: минимизация суммы искусственных переменных
        c_phase1 = np.zeros(self.n + self.m)

        # Определяем, для каких ограничений нужны искусственные переменные
        self.artificial_indices = []
        for i in range(self.m):
            if self.b[i] < -1e-10:
                # Для ограничений с отрицательной правой частью нужна искусственная переменная
                artificial_index = self.n + self.m + len(self.artificial_indices)
                self.artificial_indices.append(artificial_index)
                self.artificial_vars.append(f"a{len(self.artificial_indices)}")
                c_phase1 = np.append(c_phase1, 1.0)  # Коэффициент 1 для искусственной переменной
            else:
                c_phase1 = np.append(c_phase1, 0.0)  # Коэффициент 0, если искусственная переменная не нужна

        return c_phase1

    def _create_phase1_tableau(self, c_phase1):
        """Создание таблицы для фазы 1"""
        num_artificial = len(self.artificial_indices)
        total_vars = self.n + self.m + num_artificial

        # Создаем расширенную таблицу
        tableau = np.zeros((self.m + 1, total_vars + 1))

        # Заполняем ограничения
        artificial_counter = 0
        for i in range(self.m):
            # Основные переменные
            tableau[i, :self.n] = self.A[i]

            # Slack-переменные
            tableau[i, self.n:self.n + self.m] = np.eye(self.m)[i]

            # Искусственные переменные (только для ограничений с b_i < 0)
            if self.b[i] < -1e-10:
                tableau[i, self.n + self.m + artificial_counter] = 1.0
                artificial_counter += 1

            # Правая часть
            tableau[i, -1] = self.b[i]

        # Целевая функция фазы 1 (минимизация)
        tableau[self.m, :len(c_phase1)] = c_phase1

        # Преобразуем целевую функцию фазы 1 для исключения искусственных переменных из базиса
        for artificial_idx in self.artificial_indices:
            row_idx = artificial_idx - (self.n + self.m)
            tableau[self.m] -= tableau[row_idx]

        return tableau

    def _create_phase2_tableau(self, tableau_phase1):
        """Создание таблицы для фазы 2 из финальной таблицы фазы 1"""
        # Удаляем столбцы искусственных переменных
        phase2_cols = [i for i in range(tableau_phase1.shape[1] - 1)
                       if i not in self.artificial_indices]
        phase2_cols.append(tableau_phase1.shape[1] - 1)  # Добавляем столбец решения

        tableau_phase2 = tableau_phase1[:, phase2_cols]

        # Заменяем целевую функцию на исходную (для минимизации)
        tableau_phase2[-1, :self.n] = -self.c  # МИНУС для минимизации!
        tableau_phase2[-1, self.n:-1] = 0  # Обнуляем коэффициенты при slack-переменных

        # Исключаем базисные переменные из целевой функции
        basis = self._get_basis(tableau_phase2)
        for i, var in enumerate(basis):
            if var.startswith('x'):
                var_idx = int(var[1:]) - 1
                if var_idx < self.n:
                    multiplier = tableau_phase2[-1, var_idx]
                    tableau_phase2[-1] -= multiplier * tableau_phase2[i]

        return tableau_phase2

    def _phase2_solve(self):
        """Прямая оптимизация (фаза 2) когда начальное решение допустимо"""
        tableau = self._create_initial_tableau()
        return self._solve_phase(tableau, "Прямая оптимизация")

    def _create_initial_tableau(self):
        """Создание начальной симплекс-таблицы для МИНИМИЗАЦИИ"""
        print("\nСОЗДАНИЕ НАЧАЛЬНОЙ ТАБЛИЦЫ ДЛЯ МИНИМИЗАЦИИ:")
        print("Формат: [A | I | b]")
        print("        [-c | 0 | 0]")  # МИНУС для целевой функции!

        tableau = np.zeros((self.m + 1, self.n + self.m + 1))

        # Ограничения
        tableau[:self.m, :self.n] = self.A
        tableau[:self.m, self.n:self.n + self.m] = np.eye(self.m)
        tableau[:self.m, -1] = self.b

        # Целевая функция для минимизации: используем -c
        tableau[self.m, :self.n] = -self.c

        print(f"Размер таблицы: {tableau.shape[0]} строк × {tableau.shape[1]} столбцов")
        print("Целевая функция в таблице:", [f"{coef:8.4f}" for coef in tableau[self.m, :self.n]])
        return tableau

    def _solve_phase(self, tableau, phase_name):
        """Общий метод решения для фазы 1 или фазы 2"""
        print(f"\n{phase_name}: Начальная симплекс-таблица")
        self._print_tableau(tableau, 0, phase_name)

        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            print(f"\n{'=' * 80}")
            print(f"{phase_name}: АНАЛИЗ ИТЕРАЦИИ {iteration}")
            print(f"{'=' * 80}")

            # Проверка на оптимальность
            if self._is_optimal(tableau):
                print(f"✓ Достигнуто оптимальное решение на итерации {iteration}!")
                break

            # Выбор входящей переменной
            entering_var = self._get_entering_variable(tableau)
            if entering_var is None:
                print("✗ Задача неограничена")
                return None, None

            # Выбор выходящей переменной
            leaving_var = self._get_leaving_variable(tableau, entering_var)
            if leaving_var is None:
                print("✗ Нет допустимого выхода")
                return None, None

            iteration += 1
            print(f"\n{'=' * 80}")
            print(f"{phase_name}: ШАГ {iteration}")
            print(f"{'=' * 80}")

            # Детальный анализ выбора переменных
            self._print_pivot_analysis(tableau, entering_var, leaving_var)

            # Pivot операция
            tableau = self._pivot(tableau, leaving_var, entering_var)
            self._print_tableau(tableau, iteration, phase_name)

            if iteration == max_iterations:
                print(f"\n⚠ Достигнуто максимальное количество итераций в {phase_name}")

        # Извлечение решения
        solution = self._extract_solution(tableau)

        # Для минимизации: значение целевой функции берем с противоположным знаком
        objective_value = -self._get_objective_value(tableau)

        return solution, objective_value

    def _is_optimal(self, tableau):
        """Проверка оптимальности решения для минимизации"""
        z_row = tableau[-1, :-1]
        # Для минимизации: все коэффициенты в Z-строке должны быть <= 0
        is_optimal = np.all(z_row <= 1e-10)

        print("ПРОВЕРКА ОПТИМАЛЬНОСТИ (минимизация):")
        print("Z-строка (без правой части):", [f"{coef:8.4f}" for coef in z_row])
        print("Все коэффициенты ≤ 0?:", is_optimal)

        if not is_optimal:
            positive_count = np.sum(z_row > 1e-10)
            print(f"Положительных коэффициентов: {positive_count}")

        return is_optimal

    def _get_entering_variable(self, tableau):
        """Выбор входящей переменной для минимизации - наибольший положительный коэффициент в Z-строке"""
        z_row = tableau[-1, :-1]

        print("ВЫБОР ВХОДЯЩЕЙ ПЕРЕМЕННОЙ (минимизация):")
        print("Коэффициенты Z-строки:", [f"{coef:8.4f}" for coef in z_row])

        # Ищем положительные коэффициенты
        positive_indices = np.where(z_row > 1e-10)[0]

        if len(positive_indices) == 0:
            print("Положительных коэффициентов нет - решение оптимально")
            return None

        # Выбираем переменную с наибольшим положительным коэффициентом
        entering_var = positive_indices[np.argmax(z_row[positive_indices])]
        max_coef = z_row[entering_var]

        # Определяем имя переменной
        if entering_var < self.n:
            var_name = f"x{entering_var + 1}"
        elif entering_var < self.n + self.m:
            var_name = f"s{entering_var - self.n + 1}"
        else:
            var_name = f"a{entering_var - (self.n + self.m) + 1}"

        print(f"Выбрана переменная: {var_name}")
        print(f"Коэффициент: {max_coef:.6f}")
        print(f"Столбец: {entering_var}")

        return entering_var

    def _get_leaving_variable(self, tableau, entering_col):
        """Выбор выходящей переменной - минимальное положительное отношение"""
        print(f"\nОПРЕДЕЛЕНИЕ МАКСИМАЛЬНОГО ЗНАЧЕНИЯ для столбца {entering_col}:")

        ratios = []
        max_values = []  # Максимальные значения входящей переменной

        for i in range(self.m):
            denominator = tableau[i, entering_col]
            numerator = tableau[i, -1]

            if denominator > 1e-10:  # Положительный элемент
                max_val = numerator / denominator
                ratios.append(max_val)
                max_values.append(max_val)
                print(f"  Строка {i}: x ≤ {numerator:.4f}/{denominator:.4f} = {max_val:.4f}")
            else:
                ratios.append(np.inf)
                max_values.append(np.inf)
                print(f"  Строка {i}: x не ограничено сверху (знаменатель ≤ 0)")

        if all(r == np.inf for r in ratios):
            print("✗ Задача неограничена - переменная может расти бесконечно")
            return None

        # Максимальное значение входящей переменности = минимальное из ограничений
        leaving_var = np.argmin(ratios)
        actual_max_value = ratios[leaving_var]

        print(f"✓ Максимальное значение переменной = {actual_max_value:.6f}")
        print(f"  Определяется ограничением в строке {leaving_var}")
        print(f"  Выходящая переменная: строка {leaving_var}")

        return leaving_var

    def _print_pivot_analysis(self, tableau, entering_col, leaving_row):
        """Детальный анализ pivot-операции"""
        print("ДЕТАЛЬНЫЙ АНАЛИЗ PIVOT-ОПЕРАЦИИ:")
        print(f"Разрешающий элемент: строка {leaving_row}, столбец {entering_col}")
        print(f"Значение: {tableau[leaving_row, entering_col]:.6f}")

        # Определяем имена переменных
        if entering_col < self.n:
            entering_name = f"x{entering_col + 1}"
        elif entering_col < self.n + self.m:
            entering_name = f"s{entering_col - self.n + 1}"
        else:
            entering_name = f"a{entering_col - (self.n + self.m) + 1}"

        # Определяем выходящую переменную (базисную переменную текущей строки)
        basis = self._get_basis(tableau)
        leaving_name = basis[leaving_row] if leaving_row < len(basis) else f"строка {leaving_row}"

        print(f"Входящая переменная: {entering_name}")
        print(f"Выходящая переменная: {leaving_name}")
        print(f"Pivot: замена {leaving_name} на {entering_name} в базисе")

    def _pivot(self, tableau, leaving_row, entering_col):
        """Pivot операция - преобразование Жордана-Гаусса"""
        new_tableau = tableau.copy()

        # Разрешающий элемент
        pivot_element = new_tableau[leaving_row, entering_col]

        print(f"\nВЫПОЛНЕНИЕ PIVOT-ОПЕРАЦИИ:")
        print(f"Разрешающий элемент: {pivot_element:.6f}")

        # Нормализация разрешающей строки
        new_tableau[leaving_row] /= pivot_element
        print(f"Нормализация строки {leaving_row}: деление на {pivot_element:.6f}")

        # Обновление остальных строк
        for i in range(tableau.shape[0]):
            if i != leaving_row:
                multiplier = new_tableau[i, entering_col]
                if abs(multiplier) > 1e-10:
                    new_tableau[i] -= multiplier * new_tableau[leaving_row]
                    print(f"Обновление строки {i}: вычитание {multiplier:.6f} × строку {leaving_row}")

        return new_tableau

    def _get_basis(self, tableau):
        """Определение текущего базиса"""
        basis = []
        for j in range(tableau.shape[1] - 1):  # Все столбцы кроме решения
            col = tableau[:-1, j]
            # Проверяем, является ли столбец единичным вектором
            unit_vector_count = np.sum(np.abs(col - 1) < 1e-10)
            zero_count = np.sum(np.abs(col) < 1e-10)

            if unit_vector_count == 1 and zero_count == self.m - 1:
                row_idx = np.where(np.abs(col - 1) < 1e-10)[0][0]
                if j < self.n:
                    basis.append((row_idx, f"x{j + 1}"))
                elif j < self.n + self.m:
                    basis.append((row_idx, f"s{j - self.n + 1}"))
                else:
                    basis.append((row_idx, f"a{j - (self.n + self.m) + 1}"))

        # Сортируем по строкам
        basis.sort()
        return [var for _, var in basis]

    def _extract_solution(self, tableau):
        """Извлечение решения из финальной таблицы"""
        solution = np.zeros(self.n)
        basis = self._get_basis(tableau)

        for i, var in enumerate(basis):
            if var.startswith('x'):
                idx = int(var[1:]) - 1
                if idx < self.n:
                    solution[idx] = tableau[i, -1]

        return solution

    def _get_objective_value(self, tableau):
        """Получение значения целевой функции из таблицы"""
        return tableau[-1, -1]

    def _print_tableau(self, tableau, iteration, phase_name=""):
        """Детальный вывод симплекс-таблицы"""
        # Определяем заголовки столбцов
        headers = []
        for i in range(self.n):
            headers.append(f"x{i + 1}")
        for i in range(self.m):
            headers.append(f"s{i + 1}")

        # Добавляем искусственные переменные, если есть
        for i in range(len(self.artificial_vars)):
            headers.append(f"a{i + 1}")

        headers.append("Решение")

        basis = self._get_basis(tableau)

        title = f"СИМПЛЕКС-ТАБЛИЦА"
        if phase_name:
            title += f" ({phase_name})"
        title += f" (Итерация {iteration}):"

        print(f"\n{title}")
        print("Базисные переменные:", " | ".join(basis))

        # Вывод заголовков
        print(f"\n{'Базис':>8} ", end="")
        for header in headers:
            print(f"{header:>12} ", end="")
        print()
        print("-" * (8 + 13 * len(headers)))

        # Вывод строк ограничений
        for i in range(self.m):
            basis_var = basis[i] if i < len(basis) else "---"
            print(f"{basis_var:>8} ", end="")
            for j in range(tableau.shape[1]):
                print(f"{tableau[i, j]:12.6f} ", end="")
            print()

        # Вывод Z-строки
        print(f"{'Z':>8} ", end="")
        for j in range(tableau.shape[1]):
            print(f"{tableau[-1, j]:12.6f} ", end="")
        print()

        # Анализ текущего решения
        print(f"\nАНАЛИЗ РЕШЕНИЯ (Итерация {iteration}):")
        solution = self._extract_solution(tableau)
        for i in range(self.n):
            print(f"  x{i + 1} = {solution[i]:.8f}")

        # Для минимизации: значение целевой функции берем с противоположным знаком
        obj_val_table = self._get_objective_value(tableau)
        actual_obj_val = -obj_val_table if "Фаза 2" in phase_name or "Прямая" in phase_name else obj_val_table

        print(f"  Текущее значение целевой функции = {actual_obj_val:.8f}")

        # Проверка ограничений (только для основных переменных)
        if self.n > 0:
            print(f"ПРОВЕРКА ОГРАНИЧЕНИЙ:")
            for i in range(self.m):
                lhs = np.sum(self.A[i] * solution)
                status = "✓" if abs(lhs - self.b[i]) < 1e-6 else "✗"
                constraint_str = f"  {self.A[i, 0]:.3f}*{solution[0]:.4f}"
                for j in range(1, self.n):
                    constraint_str += f" + {self.A[i, j]:.3f}*{solution[j]:.4f}"
                constraint_str += f" = {lhs:.6f} {'≤' if self.b[i] >= 0 else '≥'} {self.b[i]:.6f} {status}"
                print(constraint_str)


def check_with_linprog(c, A, b):
    """Проверка решения с помощью scipy.optimize.linprog"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА С ПОМОЩЬЮ SCIPY.OPTIMIZE.LINPROG")
    print("=" * 80)

    try:
        # Решаем задачу минимизации
        result = linprog(c, A_ub=A, b_ub=b, method='highs')

        if result.success:
            print("✓ Решение найдено scipy.optimize.linprog:")
            for i, x_val in enumerate(result.x):
                print(f"  x{i + 1} = {x_val:.8f}")
            print(f"  Z = {result.fun:.8f}")
        else:
            print("✗ Scipy не смог найти решение:")
            print(f"  Статус: {result.message}")

        return result
    except Exception as e:
        print(f"Ошибка при вызове linprog: {e}")
        return None


def main():
    # Параметры задачи
    c = [-4.87, -3.47]
    A = [
        [6.83, 6.09],
        [0.95, 8.478]
    ]
    b = [10.97, 18.65]

    print("ЗАДАЧА ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
    print("Целевая функция: min Z = 4.87*x1 + 3.47*x2")
    print("Ограничения:")
    print("  6.83*x1 + 6.09*x2 <= 10.97")
    print("  0.95*x1 + 8.478*x2 <= 18.65")
    print("  x1 >= 0, x2 >= 0")

    # Решение симплекс-методом
    solver = SimplexSolver(c, A, b)
    solution, objective_value = solver.solve()

    # Итоговый результат
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print("=" * 80)

    if solution is not None:
        print(f"Оптимальное решение:")
        for i in range(len(solution)):
            print(f"  x{i + 1} = {solution[i]:.8f}")
        print(f"Минимальное значение Z = {objective_value:.8f}")

        # Финальная проверка ограничений
        print(f"\nФИНАЛЬНАЯ ПРОВЕРКА ОГРАНИЧЕНИЙ:")
        for i in range(len(A)):
            lhs = np.sum(A[i] * solution)
            status = "✓" if lhs <= b[i] + 1e-6 else "✗"
            constraint_str = f"  {A[i][0]:.3f}*{solution[0]:.6f} + {A[i][1]:.3f}*{solution[1]:.6f} = {lhs:.6f} <= {b[i]:.6f} {status}"
            print(constraint_str)

        # Проверка с linprog
        check_with_linprog(c, A, b)

        # Сравнение с тривиальным решением
        print(f"\nСРАВНЕНИЕ С ТРИВИАЛЬНЫМ РЕШЕНИЕМ:")
        trivial_z = np.sum(c * np.zeros(len(c)))
        print(f"  Z(0,0) = {trivial_z:.8f}")
        print(f"  Z(оптимальное) = {objective_value:.8f}")
        improvement = trivial_z - objective_value
        print(f"  Улучшение: {improvement:.8f}")

        if abs(objective_value) < 1e-6:
            print("  Тривиальное решение (0,0) является оптимальным!")
        else:
            print("  Найдено нетривиальное оптимальное решение")
    else:
        print("Решение не найдено")


if __name__ == "__main__":
    main()