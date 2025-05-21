import numpy as np
import matplotlib.pyplot as plt
# Параметры задачи
H = np.array([[6, -5], [-5, 10]])
C = np.array([10, -50])
G = np.array([[1, 1], [-1, 2], [0, -1]])
B = np.array([5, 6, 2])

# Параметры ГА (увеличены для точности)
POP_SIZE = 100
NUM_GEN = 500
MUTATION_RATE = 0.1
X_BOUNDS = [(-10, 10), (-10, 10)]
ELITISM_COUNT = 2  # кол-во лучших особей, передающихся напрямую

def objective(x):
    return x.T @ H @ x + C.T @ x

def is_feasible(x):
    return np.all(G @ x <= B)

def generate_individual():
    while True:
        x = np.array([np.random.uniform(*X_BOUNDS[0]), np.random.uniform(*X_BOUNDS[1])])
        if is_feasible(x):
            return x

def create_population():
    return [generate_individual() for _ in range(POP_SIZE)]

def roulette_selection(population, fitness):
    fitness = np.array(fitness)
    inverse_fitness = 1 / (fitness - min(fitness) + 1e-6)
    probs = inverse_fitness / inverse_fitness.sum()
    indices = np.random.choice(len(population), size=POP_SIZE, p=probs)
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child if is_feasible(child) else parent1  # возврат к родителю

def mutate(x, gen_num, max_gen):
    scale = 1.0 * (1 - gen_num / max_gen)  # адаптивная дисперсия
    x_new = x.copy()
    for i in range(len(x_new)):
        if np.random.rand() < MUTATION_RATE:
            x_new[i] += np.random.normal(0, 1) * scale
    return x_new if is_feasible(x_new) else x  # без генерации заново

def genetic_algorithm():
    population = create_population()
    best_solution = None
    best_value = float('inf')
    history = []

    for generation in range(NUM_GEN):
        fitness = [objective(x) for x in population]
        best_idx = np.argmin(fitness)

        if fitness[best_idx] < best_value:
            best_value = fitness[best_idx]
            best_solution = population[best_idx]

        history.append(best_value)

        # элитаризм
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        next_gen = sorted_population[:ELITISM_COUNT]

        selected = roulette_selection(population, fitness)

        while len(next_gen) < POP_SIZE:
            parent1 = selected[np.random.randint(POP_SIZE)]
            parent2 = selected[np.random.randint(POP_SIZE)]
            child = crossover(parent1, parent2)
            child = mutate(child, generation, NUM_GEN)
            next_gen.append(child)

        population = next_gen

    return best_solution, best_value, history

# Запуск алгоритма
solution, value, history = genetic_algorithm()
print("Лучшее решение:", solution)
print("Минимальное значение функции:", value)

# (опционально) Построим график

plt.plot(history)
plt.xlabel("Поколение")
plt.ylabel("Минимальное значение")
plt.title("Сходимость ГА")
plt.grid(True)
plt.savefig('temp_plot.png')  # Сохраняет график в файл
