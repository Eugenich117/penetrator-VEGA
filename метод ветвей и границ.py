import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linprog
import matplotlib.patches as mpatches
from collections import deque

# Исходные данные
c = np.array([-4.87, -3.47])
A = np.array([
    [6.83, 6.09],
    [0.95, 8.478]
])
b = np.array([10.97, 18.65])

print("="*70)
print("МЕТОД ВЕТВЕЙ И ГРАНИЦ - РЕШЕНИЕ ЗАДАЧИ ЦЕЛОЧИСЛЕННОГО ПРОГРАММИРОВАНИЯ")
print("="*70)
print("\nИсходная задача:")
print(f"Минимизировать: f(x) = {c[0]}*x1 + {c[1]}*x2")
print(f"При ограничениях:")
print(f"  {A[0,0]}*x1 + {A[0,1]}*x2 <= {b[0]}")
print(f"  {A[1,0]}*x1 + {A[1,1]}*x2 <= {b[1]}")
print(f"  x1, x2 >= 0")
print(f"  x1, x2 - целые числа")
print("="*70)

class BranchAndBoundNode:
    """Класс для представления узла в дереве ветвления"""
    node_counter = 0

    def __init__(self, lower_bounds, upper_bounds, parent_id=None, branch_var=None, branch_dir=None):
        BranchAndBoundNode.node_counter += 1
        self.id = BranchAndBoundNode.node_counter
        self.lower_bounds = lower_bounds.copy()
        self.upper_bounds = upper_bounds.copy()
        self.parent_id = parent_id
        self.branch_var = branch_var
        self.branch_dir = branch_dir
        self.solution = None
        self.objective = None
        self.is_feasible = False
        self.is_integer = False
        self.is_pruned = False
        self.prune_reason = None

def solve_lp_relaxation(c, A, b, lower_bounds, upper_bounds):
    """Решение задачи линейного программирования симплекс-методом"""
    n = len(c)
    bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(n)]
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='simplex')
    return result

def is_integer_solution(x, tol=1e-6):
    """Проверка, является ли решение целочисленным"""
    return np.all(np.abs(x - np.round(x)) < tol)

def branch_and_bound(c, A, b, verbose=True):
    """Реализация метода ветвей и границ"""
    n = len(c)

    initial_lower = np.zeros(n)
    initial_upper = np.full(n, np.inf)

    queue = deque()
    root = BranchAndBoundNode(initial_lower, initial_upper)
    queue.append(root)

    best_integer_solution = None
    best_integer_objective = np.inf
    all_nodes = []

    if verbose:
        print("\n" + "="*70)
        print("НАЧАЛО РАБОТЫ АЛГОРИТМА")
        print("="*70)

    while queue:
        node = queue.popleft()

        if verbose:
            print(f"\n{'='*70}")
            print(f"ОБРАБОТКА УЗЛА {node.id}")
            print(f"{'='*70}")

            if node.parent_id:
                print(f"Родительский узел: {node.parent_id}")
                if node.branch_var is not None:
                    direction = "≥" if node.branch_dir == "up" else "≤"
                    bound_value = node.lower_bounds[node.branch_var] if node.branch_dir == "up" else node.upper_bounds[node.branch_var]
                    print(f"Ветвление: x{node.branch_var+1} {direction} {bound_value}")

            print(f"Границы переменных:")
            for i in range(n):
                lb = node.lower_bounds[i] if node.lower_bounds[i] != -np.inf else "-∞"
                ub = node.upper_bounds[i] if node.upper_bounds[i] != np.inf else "+∞"
                print(f"  {lb} <= x{i+1} <= {ub}")

        result = solve_lp_relaxation(c, A, b, node.lower_bounds, node.upper_bounds)

        if not result.success:
            if verbose:
                print(f"❌ Симплекс-метод: решение не найдено (недопустимая область)")
            node.is_pruned = True
            node.prune_reason = "infeasible"
            all_nodes.append(node)
            continue

        node.solution = result.x
        node.objective = result.fun
        node.is_feasible = True

        if verbose:
            print(f"✓ Симплекс-метод: задача решена успешно")
            print(f"  Решение: x1 = {node.solution[0]:.6f}, x2 = {node.solution[1]:.6f}")
            print(f"  Значение целевой функции: f = {node.objective:.6f}")

        if node.objective >= best_integer_objective:
            if verbose:
                print(f"✂ Отсечение по границе: f = {node.objective:.6f} >= {best_integer_objective:.6f}")
            node.is_pruned = True
            node.prune_reason = "bound"
            all_nodes.append(node)
            continue

        if is_integer_solution(node.solution):
            if verbose:
                print(f"✓ Решение является целочисленным!")
            node.is_integer = True

            if node.objective < best_integer_objective:
                best_integer_solution = node.solution.copy()
                best_integer_objective = node.objective
                if verbose:
                    print(f"★ Новое лучшее целочисленное решение!")
                    print(f"  x* = ({int(node.solution[0])}, {int(node.solution[1])})")
                    print(f"  f* = {best_integer_objective:.6f}")

            all_nodes.append(node)
            continue

        fractional_parts = np.abs(node.solution - np.round(node.solution))
        branch_var = np.argmax(fractional_parts)
        branch_value = node.solution[branch_var]

        if verbose:
            print(f"\n→ Решение не целочисленное. Ветвление по переменной x{branch_var+1}")
            print(f"  Текущее значение: x{branch_var+1} = {branch_value:.6f}")
            print(f"  Создаем две ветви: x{branch_var+1} ≤ {int(np.floor(branch_value))} и x{branch_var+1} ≥ {int(np.ceil(branch_value))}")

        left_lower = node.lower_bounds.copy()
        left_upper = node.upper_bounds.copy()
        left_upper[branch_var] = np.floor(branch_value)
        left_child = BranchAndBoundNode(left_lower, left_upper, node.id, branch_var, "down")
        queue.append(left_child)

        right_lower = node.lower_bounds.copy()
        right_upper = node.upper_bounds.copy()
        right_lower[branch_var] = np.ceil(branch_value)
        right_child = BranchAndBoundNode(right_lower, right_upper, node.id, branch_var, "up")
        queue.append(right_child)

        all_nodes.append(node)

    if verbose:
        print(f"\n{'='*70}")
        print("АЛГОРИТМ ЗАВЕРШЕН")
        print(f"{'='*70}")

        if best_integer_solution is not None:
            print(f"\n★ ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО:")
            print(f"  x1 = {int(best_integer_solution[0])}")
            print(f"  x2 = {int(best_integer_solution[1])}")
            print(f"  f(x*) = {best_integer_objective:.6f}")
            print(f"\nПроверка ограничений:")
            for i in range(len(b)):
                lhs = A[i] @ best_integer_solution
                print(f"  Ограничение {i+1}: {lhs:.4f} <= {b[i]:.4f} {'✓' if lhs <= b[i] + 1e-6 else '✗'}")
        else:
            print("\n✗ Целочисленное решение не найдено")

        print(f"\nВсего обработано узлов: {len(all_nodes)}")
        print(f"На каждом узле применялся симплекс-метод для решения задачи ЛП")

    return best_integer_solution, best_integer_objective, all_nodes

# Запускаем метод ветвей и границ
optimal_solution, optimal_objective, all_nodes = branch_and_bound(c, A, b)

# ============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "="*70)
print("ПОСТРОЕНИЕ ГРАФИКОВ")
print("="*70)

# Создаем сетку для построения графиков
x1 = np.linspace(-0.5, 3, 100)
x2 = np.linspace(-0.5, 3, 100)
X1, X2 = np.meshgrid(x1, x2)

# Целевая функция
Z = c[0] * X1 + c[1] * X2

# Создаем фигуру
fig = plt.figure(figsize=(20, 6))

# ========== 3D График ==========
ax1 = fig.add_subplot(131, projection='3d')

# Поверхность целевой функции
surf = ax1.plot_surface(X1, X2, Z, alpha=0.5, cmap='viridis', edgecolor='none')

# Отображаем все узлы с решениями
for node in all_nodes:
    if node.solution is not None:
        px1, px2 = node.solution[0], node.solution[1]
        pz = node.objective

        if node.is_integer and node.objective == optimal_objective:
            # Оптимальное решение
            ax1.scatter([px1], [px2], [pz], color='red', s=300, marker='*',
                       edgecolors='black', linewidth=2, zorder=10, label='Оптимум')
        elif node.is_integer:
            # Другое целочисленное решение
            ax1.scatter([px1], [px2], [pz], color='lightgreen', s=100,
                       edgecolors='black', linewidth=1.5, zorder=5)
        elif node.is_pruned:
            # Отсеченный узел
            ax1.scatter([px1], [px2], [pz], color='lightcoral', s=60,
                       edgecolors='black', linewidth=1, zorder=4, alpha=0.6)
        else:
            # Промежуточный узел
            ax1.scatter([px1], [px2], [pz], color='yellow', s=80,
                       edgecolors='black', linewidth=1.5, zorder=5)

ax1.set_xlabel('x₁', fontsize=12, fontweight='bold')
ax1.set_ylabel('x₂', fontsize=12, fontweight='bold')
ax1.set_zlabel('f(x₁, x₂)', fontsize=12, fontweight='bold')
ax1.set_title('3D: Целевая функция и узлы дерева\nf = -4.87x₁ - 3.47x₂', fontsize=13, fontweight='bold')
handles, labels = ax1.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax1.legend(*zip(*unique), loc='best')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# ========== 2D График ==========
ax2 = fig.add_subplot(132)

# Линии ограничений
x1_line = np.linspace(0, 3, 200)

# Ограничение 1
x2_constraint1 = (b[0] - A[0, 0] * x1_line) / A[0, 1]
ax2.plot(x1_line, x2_constraint1, 'b-', linewidth=2,
         label=f'{A[0,0]:.2f}x₁ + {A[0,1]:.2f}x₂ ≤ {b[0]:.2f}')
ax2.fill_between(x1_line, 0, x2_constraint1, where=(x2_constraint1 >= 0),
                 alpha=0.15, color='blue')

# Ограничение 2
x2_constraint2 = (b[1] - A[1, 0] * x1_line) / A[1, 1]
ax2.plot(x1_line, x2_constraint2, 'g-', linewidth=2,
         label=f'{A[1,0]:.2f}x₁ + {A[1,1]:.2f}x₂ ≤ {b[1]:.2f}')
ax2.fill_between(x1_line, 0, x2_constraint2, where=(x2_constraint2 >= 0),
                 alpha=0.15, color='green')

# Допустимая область
x2_feasible = np.minimum(x2_constraint1, x2_constraint2)
x2_feasible = np.maximum(x2_feasible, 0)
ax2.fill_between(x1_line, 0, x2_feasible,
                 where=(x2_feasible >= 0) & (x1_line >= 0),
                 alpha=0.3, color='yellow', label='Допустимая область')

# Линии уровня целевой функции
levels = np.linspace(-8, 0, 15)
contour = ax2.contour(X1, X2, Z, levels=levels, colors='gray',
                      alpha=0.4, linestyles='dashed', linewidths=0.8)
ax2.clabel(contour, inline=True, fontsize=7)

# Целочисленные точки в допустимой области
for i in range(0, 4):
    for j in range(0, 4):
        if (A[0, 0] * i + A[0, 1] * j <= b[0]) and (A[1, 0] * i + A[1, 1] * j <= b[1]):
            ax2.plot(i, j, 's', color='lightblue', markersize=10,
                    markeredgecolor='navy', markeredgewidth=1.5, alpha=0.7)

# Отображаем узлы дерева
node_colors = {
    'optimal': ('red', 20, 250),
    'integer': ('lightgreen', 12, 120),
    'feasible': ('yellow', 10, 100),
    'infeasible': ('lightcoral', 8, 70)
}

for i, node in enumerate(all_nodes):
    if node.solution is not None:
        px1, px2 = node.solution[0], node.solution[1]

        if node.is_integer and node.objective == optimal_objective:
            color, marker_size, z = node_colors['optimal']
            ax2.plot(px1, px2, '*', color=color, markersize=marker_size,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
        elif node.is_integer:
            color, marker_size, z = node_colors['integer']
            ax2.plot(px1, px2, 'o', color=color, markersize=marker_size,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=7)
        else:
            color, marker_size, z = node_colors['feasible']
            ax2.plot(px1, px2, 'o', color=color, markersize=marker_size,
                    markeredgecolor='black', markeredgewidth=1, zorder=5, alpha=0.7)

        # Номер узла
        ax2.annotate(f'{node.id}', (px1, px2), xytext=(3, 3),
                    textcoords='offset points', fontsize=8, fontweight='bold')

# Оптимум с подписью
if optimal_solution is not None:
    opt_x1, opt_x2 = int(optimal_solution[0]), int(optimal_solution[1])
    ax2.plot(opt_x1, opt_x2, '*', color='red', markersize=22,
            markeredgecolor='black', markeredgewidth=2.5,
            label=f'Оптимум: ({opt_x1}, {opt_x2}), f={optimal_objective:.2f}', zorder=11)

ax2.set_xlabel('x₁', fontsize=12, fontweight='bold')
ax2.set_ylabel('x₂', fontsize=12, fontweight='bold')
ax2.set_title('2D: Допустимая область и узлы дерева\nветвей и границ',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(-0.3, 2.5)
ax2.set_ylim(-0.3, 2.5)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# ========== Дерево ветвлений (ИСПРАВЛЕННАЯ ЧАСТЬ) ==========
ax3 = fig.add_subplot(133)
ax3.axis('off')

def build_tree_structure(all_nodes):
    """Построение структуры дерева из списка узлов"""
    nodes_dict = {node.id: node for node in all_nodes}

    # Строим дерево
    def build_node_dict(node):
        children = [n for n in all_nodes if n.parent_id == node.id]

        status = ""
        if node.is_integer:
            if node.objective == optimal_objective:
                color = 'gold'
                status = "★"
            else:
                color = 'lightgreen'
                status = "✓"
        elif node.is_pruned:
            if node.prune_reason == "infeasible":
                color = 'lightcoral'
                status = "✗"
            else:
                color = 'lightyellow'
                status = "✂"
        else:
            color = 'lightblue'
            status = ""

        label = f'Узел {node.id} {status}\n'
        if node.solution is not None:
            label += f'({node.solution[0]:.2f}, {node.solution[1]:.2f})\n'
            label += f'f={node.objective:.2f}'
        else:
            label += 'недопустимо'

        return {
            'id': node.id,
            'label': label,
            'color': color,
            'children': [build_node_dict(c) for c in sorted(children, key=lambda x: x.id)]
        }

    root = [n for n in all_nodes if n.parent_id is None][0]
    return build_node_dict(root)

def layout_tree(node, x=0.5, y=1.0, level=0, width=1.0):
    """Рекурсивная расстановка узлов дерева с уменьшенным расстоянием между ветвями"""
    # Уменьшенные вертикальные расстояния между уровнями
    level_heights = [0.20, 0.18, 0.16, 0.14, 0.12]
    level_height = level_heights[min(level, len(level_heights)-1)]

    node['x'] = x
    node['y'] = y

    if not node['children']:
        return

    n_children = len(node['children'])

    if n_children == 1:
        child_positions = [x]
        child_widths = [width * 0.7]  # Уменьшена ширина для одного ребенка
    else:
        # Уменьшенное горизонтальное расстояние между детьми
        spacing_factor = 1.6  # Было 1.8, уменьшили
        total_width = width * spacing_factor

        if n_children == 2:
            # Для двух детей делаем их ближе друг к другу
            offset = total_width / 3.5  # Увеличен делитель для уменьшения расстояния
            child_positions = [x - offset, x + offset]
            child_widths = [total_width / 2.8] * 2  # Уменьшены ширины
        else:
            step = total_width / (n_children - 1)
            child_positions = [x - total_width/2 + i * step for i in range(n_children)]
            child_widths = [total_width / (n_children * 1.3)] * n_children

    for child, child_x, child_width in zip(node['children'], child_positions, child_widths):
        child_y = y - level_height
        layout_tree(child, child_x, child_y, level + 1, child_width)

def draw_tree(ax, node, parent_x=None, parent_y=None):
    """Рисование дерева с рамками для всех узлов"""
    x, y = node['x'], node['y']

    # Рисуем связь с родителем
    if parent_x is not None:
        ax.plot([parent_x, x], [parent_y, y], 'k-', linewidth=1.5, alpha=0.6, zorder=1)

    # Определяем размеры и стиль рамки в зависимости от типа узла
    if node['color'] == 'gold':
        # Оптимальное решение
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=node['color'],
                         edgecolor='red', linewidth=3, alpha=1.0)
        fontsize = 7
        bbox = dict(boxstyle="round,pad=0.3", facecolor=node['color'],
                   edgecolor='red', linewidth=3)
    elif node['color'] == 'lightcoral':
        # Недопустимый узел
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor=node['color'],
                         edgecolor='darkred', linewidth=2, alpha=0.9)
        fontsize = 7
        bbox = dict(boxstyle="round,pad=0.2", facecolor=node['color'],
                   edgecolor='darkred', linewidth=2)
    elif node['color'] == 'lightgreen':
        # Целочисленное решение
        bbox_props = dict(boxstyle="round,pad=0.25", facecolor=node['color'],
                         edgecolor='darkgreen', linewidth=2, alpha=1.0)
        fontsize = 7
        bbox = dict(boxstyle="round,pad=0.25", facecolor=node['color'],
                   edgecolor='darkgreen', linewidth=2)
    elif node['color'] == 'lightyellow':
        # Отсеченный узел
        bbox_props = dict(boxstyle="round,pad=0.25", facecolor=node['color'],
                         edgecolor='orange', linewidth=2, alpha=0.9)
        fontsize = 7
        bbox = dict(boxstyle="round,pad=0.25", facecolor=node['color'],
                   edgecolor='orange', linewidth=2)
    else:
        # Обычный узел
        bbox_props = dict(boxstyle="round,pad=0.25", facecolor=node['color'],
                         edgecolor='darkblue', linewidth=2, alpha=1.0)
        fontsize = 7
        bbox = dict(boxstyle="round,pad=0.25", facecolor=node['color'],
                   edgecolor='darkblue', linewidth=2)

    # Добавляем текст с рамкой
    ax.text(x, y, node['label'], ha='center', va='center',
            fontsize=fontsize, fontweight='bold', bbox=bbox, zorder=3)

    # Рисуем детей
    for child in node.get('children', []):
        draw_tree(ax, child, x, y)

# Строим и рисуем дерево
tree_root = build_tree_structure(all_nodes)
layout_tree(tree_root, x=0.5, y=0.95, level=0, width=0.8)  # Уменьшена начальная ширина
draw_tree(ax3, tree_root)

ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(-0.05, 1.05)
ax3.set_title('Дерево ветвей и границ\n',
              fontsize=12, fontweight='bold', pad=15)

# Легенда с улучшенным отображением
legend_elements = [
    plt.Rectangle((0,0), 1, 1, facecolor='lightblue', edgecolor='darkblue',
                  linewidth=2, label='Решение симплекс-методом'),
    plt.Rectangle((0,0), 1, 1, facecolor='lightcoral', edgecolor='darkred',
                  linewidth=2, label='Недопустимый узел'),
    plt.Rectangle((0,0), 1, 1, facecolor='lightyellow', edgecolor='orange',
                  linewidth=2, label='Отсечено по границе'),
    plt.Rectangle((0,0), 1, 1, facecolor='lightgreen', edgecolor='darkgreen',
                  linewidth=2, label='Целочисленное решение'),
    plt.Rectangle((0,0), 1, 1, facecolor='gold', edgecolor='red',
                  linewidth=3, label='Оптимум')
]
ax3.legend(handles=legend_elements, loc='lower center', fontsize=8, ncol=2,
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('branch_and_bound_solution.png', dpi=150, bbox_inches='tight')
plt.show()

# Итоговая сводка
print("\n" + "="*70)
print("СВОДКА ПО ВСЕМ УЗЛАМ")
print("="*70)
print(f"{'Узел':<6} {'Статус':<30} {'Решение (x1, x2)':<20} {'f(x)':<12}")
print("-"*70)
for node in all_nodes:
    status = ""
    if node.is_integer:
        if node.objective == optimal_objective:
            status = "★ Оптимум"
        else:
            status = "✓ Целочисленное"
    elif node.is_pruned:
        if node.prune_reason == "infeasible":
            status = "✗ Недопустимо"
        elif node.prune_reason == "bound":
            status = "✂ Отсечено по границе"
    else:
        status = "→ Ветвление (симплекс-метод)"

    sol_str = f"({node.solution[0]:.4f}, {node.solution[1]:.4f})" if node.solution is not None else "—"
    obj_str = f"{node.objective:.6f}" if node.objective is not None else "—"

    print(f"{node.id:<6} {status:<30} {sol_str:<20} {obj_str:<12}")
