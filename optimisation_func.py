import numpy as np

# Параметры задачи
H = np.array([[6, -5], [-5, 10]])
C = np.array([10, -50])
G = np.array([[1, 1], [-1, 2], [0, -1]])
B = np.array([5, 6, 2])

# Параметры ГА
POP_SIZE = 50
NUM_GEN = 100
MUTATION_RATE = 0.1
X_BOUNDS = [(-10, 10), (-10, 10)]

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


"""рулеточный отбор для увеличения эффективности отбора поколений"""
def roulette_selection(population, fitness):
    fitness = np.array(fitness)
    inverse_fitness = 1 / (fitness - min(fitness) + 1e-6)  # избегаем деления на 0
    probs = inverse_fitness / inverse_fitness.sum()
    indices = np.random.choice(len(population), size=POP_SIZE, p=probs)
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child if is_feasible(child) else generate_individual()

def mutate(x):
    for i in range(len(x)):
        if np.random.rand() < MUTATION_RATE:
            x[i] += np.random.normal(0, 1)
    return x if is_feasible(x) else generate_individual()

def genetic_algorithm():
    population = create_population()
    best_solution = None
    best_value = float('inf')

    for generation in range(NUM_GEN):
        fitness = [objective(x) for x in population]
        best_idx = np.argmin(fitness)

        if fitness[best_idx] < best_value:
            best_value = fitness[best_idx]
            best_solution = population[best_idx]

        selected = roulette_selection(population, fitness)
        next_gen = []

        for i in range(0, POP_SIZE, 2):
            parent1, parent2 = selected[i], selected[(i+1)%POP_SIZE]
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            next_gen.extend([child1, child2])

        population = next_gen[:POP_SIZE]

    return best_solution, best_value

# Запуск алгоритма
solution, value = genetic_algorithm()
print("Лучшее решение:", solution)
print("Минимальное значение функции:", value)
