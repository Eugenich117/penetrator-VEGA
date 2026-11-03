# Импорт необходимых библиотек
import numpy as np  # для работы с матрицами и векторами
import matplotlib.pyplot as plt  # для визуализации результатов

# ==================== ПАРАМЕТРЫ ЗАДАЧИ ====================
# Матрица квадратичной формы (2x2)
H = np.array([[6, -5], [-5, 10]])
# Вектор линейных коэффициентов
C = np.array([10, -50])
# Матрица коэффициентов ограничений (3 ограничения x 2 переменные)
G = np.array([[1, 1], [-1, 2], [0, -1]])
# Правая часть ограничений
B = np.array([5, 6, 2])

# ==================== ПАРАМЕТРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ====================
POP_SIZE = 100  # Размер популяции
NUM_GEN = 500  # Количество поколений
MUTATION_RATE = 0.8  # Вероятность мутации
X_BOUNDS = [(-10, 10), (-10, 10)]  # Границы для переменных
ELITISM_COUNT = 2  # Количество лучших особей, переходящих в следующее поколение без изменений


# ==================== ОПРЕДЕЛЕНИЕ ФУНКЦИЙ ====================

def objective(x):
    """Целевая функция: xᵀHx + Cᵀx"""
    return x.T @ H @ x + C.T @ x


def is_feasible(x):
    """Проверка выполнения ограничений Gx ≤ B"""
    return np.all(G @ x <= B)


def generate_individual():
    """Генерация одной допустимой особи (решения)"""
    while True:
        # Генерируем случайную точку в заданных границах
        x = np.array([np.random.uniform(*X_BOUNDS[0]),
                      np.random.uniform(*X_BOUNDS[1])])
        # Повторяем, пока не найдем допустимое решение
        if is_feasible(x):
            return x


def create_population():
    """Создание начальной популяции"""
    return [generate_individual() for _ in range(POP_SIZE)]


def roulette_selection(population, fitness):
    """Селекция методом рулетки с вероятностями, обратно пропорциональными значению функции"""
    fitness = np.array(fitness)
    # Инвертируем значения приспособленности (так как мы минимизируем)
    inverse_fitness = 1 / (fitness - min(fitness) + 1e-6)  # +1e-6 чтобы избежать деления на 0
    # Нормализуем в вероятности
    probs = inverse_fitness / inverse_fitness.sum()
    # Выбираем индексы особей согласно вероятностям
    indices = np.random.choice(len(population), size=POP_SIZE, p=probs)
    return [population[i] for i in indices]


def crossover(parent1, parent2):
    """Аффинное скрещивание: линейная комбинация двух родителей"""
    alpha = np.random.rand()  # случайный вес
    child = alpha * parent1 + (1 - alpha) * parent2
    # Если потомок недопустим, возвращаем первого родителя
    return child if is_feasible(child) else parent1


def mutate(x, gen_num, max_gen):
    """Адаптивная мутация с уменьшающимся воздействием"""
    # Масштаб мутации уменьшается с номером поколения
    scale = 1.0 * (1 - gen_num / max_gen)
    x_new = x.copy()
    for i in range(len(x_new)):
        # Применяем мутацию с заданной вероятностью
        if np.random.rand() < MUTATION_RATE:
            # Добавляем гауссовский шум
            x_new[i] += np.random.normal(0, 1) * scale
    # Возвращаем мутанта, если он допустим, иначе исходную особь
    return x_new if is_feasible(x_new) else x


def genetic_algorithm():
    """Основная функция генетического алгоритма"""
    # Вычисление безусловного минимума (аналитически)
    H_inv = np.linalg.inv(H)
    unconditional_min = -np.dot(H_inv, C)  # x = -H^(-1)C
    unconditional_value = objective(unconditional_min)
    # Инициализация популяции
    population = create_population()
    best_solution = None
    best_value = float('inf')
    history = []  # для хранения истории лучших значений

    # Основной цикл по поколениям
    for generation in range(NUM_GEN):
        # Вычисляем значение функции для всех особей
        fitness = [objective(x) for x in population]
        # Находим индекс лучшей особи
        best_idx = np.argmin(fitness)

        # Обновляем лучшее решение
        if fitness[best_idx] < best_value:
            best_value = fitness[best_idx]
            best_solution = population[best_idx]

        # Сохраняем историю для визуализации
        history.append(best_value)

        # Присваивание баллов (например, на основе близости к безусловному минимуму)
        # Баллы = 100 - (расстояние до безусловного минимума * весовой коэффициент)
        scores = [100 - np.abs(objective(x) - unconditional_value) * 10 for x in population]
        # Ограничиваем баллы диапазоном [0, 100]
        scores = np.clip(scores, 0, 100)

        # Выводим информацию о лучшей особи и её балле (для отладки)
        print(f"Generation {generation}, Best Value: {best_value}, Score: {scores[best_idx]}")

        # ========== ФОРМИРОВАНИЕ НОВОГО ПОКОЛЕНИЯ ==========
        # Элитизм: выбираем лучших особей без изменений
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        next_gen = sorted_population[:ELITISM_COUNT]

        # Селекция: выбираем родителей методом рулетки
        selected = roulette_selection(population, fitness)

        # Скрещивание и мутация до заполнения популяции
        while len(next_gen) < POP_SIZE:
            parent1 = selected[np.random.randint(POP_SIZE)]
            parent2 = selected[np.random.randint(POP_SIZE)]
            child = crossover(parent1, parent2)
            child = mutate(child, generation, NUM_GEN)
            next_gen.append(child)

        # Переходим к новому поколению
        population = next_gen

    return best_solution, best_value, history


# ==================== ЗАПУСК АЛГОРИТМА ====================
solution, value, history = genetic_algorithm()
print("Лучшее решение:", solution)
print("Минимальное значение функции:", value)

# ==================== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================
plt.plot(history)
plt.xlabel("Поколение")
plt.ylabel("Минимальное значение")
plt.title("Сходимость генетического алгоритма")
plt.grid(True)
plt.savefig('ga_convergence.png')  # Сохранение графика в файл
plt.show()