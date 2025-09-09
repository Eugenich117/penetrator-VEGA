import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class AircraftMaintenanceCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Калькулятор времени восстановления воздушных судов - МиГ-29")
        self.root.geometry("1400x900")

        self.create_widgets()

    def create_widgets(self):
        # Создаем notebook для разделения на вкладки
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Вкладка 1: Расчет по экспериментальным данным
        frame_experimental = ttk.Frame(notebook)
        notebook.add(frame_experimental, text="Экспериментальные данные")
        self.create_experimental_tab(frame_experimental)

        # Вкладка 2: Расчет на стадии проектирования
        frame_design = ttk.Frame(notebook)
        notebook.add(frame_design, text="Проектирование")
        self.create_design_tab(frame_design)

        # Вкладка 3: Справочная информация
        frame_reference = ttk.Frame(notebook)
        notebook.add(frame_reference, text="Справочник")
        self.create_reference_tab(frame_reference)

    def create_experimental_tab(self, parent):
        # Ввод данных о времени восстановления
        input_frame = ttk.LabelFrame(parent, text="Ввод данных о времени восстановления (часы)")
        input_frame.pack(fill='x', padx=10, pady=5)

        # Поле для ввода времени
        ttk.Label(input_frame, text="Введите время восстановления через запятую:").pack(pady=5)
        self.time_entry = ttk.Entry(input_frame, width=100)
        self.time_entry.pack(pady=5)
        self.time_entry.insert(0,
                               "2.5, 3.0, 4.2, 1.8, 5.5, 3.8, 2.2, 6.0, 4.5, 3.2, 2.8, 5.2, 3.5, 4.8, 2.0, 6.5, 3.8, 4.0, 2.5, 5.8, 3.2, 4.5, 2.8, 6.2, 3.5")

        # Уровень доверия
        ttk.Label(input_frame, text="Доверительная вероятность β:").pack(pady=5)
        self.confidence_var = tk.StringVar(value="0.95")
        confidence_combo = ttk.Combobox(input_frame, textvariable=self.confidence_var,
                                        values=["0.80", "0.90", "0.95", "0.975", "0.990", "0.995", "0.9975", "0.999"])
        confidence_combo.pack(pady=5)

        # Кнопка расчета
        ttk.Button(input_frame, text="Рассчитать", command=self.calculate_experimental).pack(pady=10)

        # Результаты
        result_frame = ttk.LabelFrame(parent, text="Результаты расчета для МиГ-29")
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=15)
        self.result_text.pack(fill='both', expand=True, padx=5, pady=5)

        # График
        self.fig_experimental = Figure(figsize=(10, 5))
        self.canvas_experimental = FigureCanvasTkAgg(self.fig_experimental, result_frame)
        self.canvas_experimental.get_tk_widget().pack(fill='both', expand=True)

    def create_design_tab(self, parent):
        # Таблица для ввода данных о конструктивных элементах
        table_frame = ttk.LabelFrame(parent, text="Конструктивные элементы МиГ-29")
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Создаем таблицу
        columns = ("Компонент", "Количество", "Интенсивность отказов (10⁻⁵/ч)", "Время восстановления (ч)")
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180)

        # Добавляем scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Кнопки для управления таблицей
        button_frame = ttk.Frame(table_frame)
        button_frame.pack(fill='x', pady=5)

        ttk.Button(button_frame, text="Добавить элемент", command=self.add_element).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Удалить элемент", command=self.delete_element).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Загрузить типовые данные", command=self.load_mig29_data).pack(side=tk.LEFT,
                                                                                                     padx=5)

        # Кнопка расчета
        ttk.Button(parent, text="Рассчитать время восстановления", command=self.calculate_design).pack(pady=10)

        # Результаты
        result_frame = ttk.LabelFrame(parent, text="Результаты проектирования для МиГ-29")
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.design_result_text = scrolledtext.ScrolledText(result_frame, height=12)
        self.design_result_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Загружаем типовые данные МиГ-29
        self.load_mig29_data()

    def create_reference_tab(self, parent):
        # Справочная информация
        info_text = """
        МЕТОДИКА РАСЧЕТА ВРЕМЕНИ ВОССТАНОВЛЕНИЯ МИГ-29

        СТАТИСТИКА ОТКАЗОВ ДЛЯ МИГ-29 (примерные данные):

        Компонент                | Инт. отказов (10⁻⁵/ч) | Время восстановления (ч)
        ------------------------|----------------------|-------------------------
        Двигатель РД-33          | 1.2-2.0              | 4-8
        Система управления       | 0.8-1.5              | 2-4
        Радиолокационная станция | 1.5-2.5              | 3-6
        Система навигации        | 0.6-1.2              | 2-3
        Гидравлическая система   | 0.9-1.8              | 2-5
        Топливная система        | 0.7-1.4              | 1-3
        Электрооборудование      | 1.0-2.0              | 1-4
        Шасси                    | 0.4-0.8              | 3-6

        ФОРМУЛЫ РАСЧЕТА:

        1. Для экспериментальных данных:
           t_B = (1/n) * Σ τ_B_i   (среднее арифметическое)

        2. Доверительные границы (логарифмически нормальное распределение):
           lg t_BH = â + 1.51S² - u_γ * (s/√n) * √(1 + 2.65S²)
           lg t_BB = â + 1.51S² + u_γ * (s/√n) * √(1 + 2.65S²)

        3. Для проектирования:
           t_Bη = Σ P_ξ * t_Bξ^э
           t_B = Σ P_η * t_Bη

        КВАНИТИЛИ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ:
          β     | 0.80 | 0.90 | 0.95 | 0.975 | 0.990 | 0.995 | 0.9975 | 0.999
          u_γ   | 0.842|1.282 |1.645 | 1.960 | 2.326 | 2.576 | 2.807  | 3.090

        ОПЕРАТИВНОЕ ВРЕМЯ ВОССТАНОВЛЕНИЯ ВКЛЮЧАЕТ:
          - Обнаружение отказа (15-30% времени)
          - Устранение неисправности (40-60% времени)
          - Регулировка и настройка (10-20% времени)
          - Проверка работоспособности (5-15% времени)

        СРЕДНЕЕ ВРЕМЯ ВОССТАНОВЛЕНИЯ МИГ-29: 
          По эксплуатационным данным: 3.5-6.5 часов
          На стадии проектирования: 4.2-5.8 часов
        """

        text_widget = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        text_widget.insert(1.0, info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)

    def load_mig29_data(self):
        # Очищаем текущие данные
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Типовые данные для МиГ-29 (основаны на реальной статистике)
        mig29_components = [
            ("Двигатель РД-33", "2", "1.6", "6.0"),
            ("Система управления", "1", "1.2", "3.2"),
            ("Радиолокационная станция", "1", "2.0", "4.5"),
            ("Система навигации", "1", "0.9", "2.5"),
            ("Гидравлическая система", "1", "1.4", "3.8"),
            ("Топливная система", "1", "1.1", "2.2"),
            ("Электрооборудование", "1", "1.5", "3.5"),
            ("Шасси", "3", "0.6", "4.2"),
            ("Тормозная система", "1", "0.8", "2.8"),
            ("Система кондиционирования", "1", "0.7", "2.0"),
            ("Приборная панель", "1", "0.5", "1.8"),
            ("Система вооружения", "1", "1.8", "5.2")
        ]

        for component in mig29_components:
            self.tree.insert("", tk.END, values=component)

    def add_element(self):
        # Диалог для добавления нового элемента
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавить конструктивный элемент")
        dialog.geometry("400x250")

        ttk.Label(dialog, text="Компонент:").pack(pady=5)
        type_entry = ttk.Entry(dialog)
        type_entry.pack(pady=5)

        ttk.Label(dialog, text="Количество:").pack(pady=5)
        count_entry = ttk.Entry(dialog)
        count_entry.pack(pady=5)

        ttk.Label(dialog, text="Интенсивность отказов (10⁻⁵/ч):").pack(pady=5)
        lambda_entry = ttk.Entry(dialog)
        lambda_entry.pack(pady=5)

        ttk.Label(dialog, text="Время восстановления (ч):").pack(pady=5)
        time_entry = ttk.Entry(dialog)
        time_entry.pack(pady=5)

        def add():
            values = (
                type_entry.get(),
                count_entry.get(),
                lambda_entry.get(),
                time_entry.get()
            )
            if all(values):
                self.tree.insert("", tk.END, values=values)
                dialog.destroy()

        ttk.Button(dialog, text="Добавить", command=add).pack(pady=10)

    def delete_element(self):
        selected = self.tree.selection()
        if selected:
            self.tree.delete(selected)

    def calculate_experimental(self):
        try:
            # Получаем данные из поля ввода
            times_str = self.time_entry.get()
            times = [float(x.strip()) for x in times_str.split(',') if x.strip()]

            if not times:
                messagebox.showerror("Ошибка", "Введите данные о времени восстановления")
                return

            n = len(times)
            t_B = sum(times) / n  # Формула (1)

            # Вычисляем параметры для доверительных интервалов
            log_times = [math.log10(t) for t in times]
            a_hat = sum(log_times) / n  # Формула (4)

            S_squared = sum((x - a_hat) ** 2 for x in log_times) / (n - 1)  # Формула (5)
            S = math.sqrt(S_squared)

            # Получаем квантиль для выбранного уровня доверия
            beta = float(self.confidence_var.get())
            gamma = (beta + 1) / 2

            u_gamma_dict = {
                0.80: 0.842, 0.90: 1.282, 0.95: 1.645, 0.975: 1.960,
                0.990: 2.326, 0.995: 2.576, 0.9975: 2.807, 0.999: 3.090
            }
            u_gamma = u_gamma_dict.get(beta, 1.645)

            # Вычисляем доверительные границы (формулы 2 и 3)
            term = u_gamma * (S / math.sqrt(n)) * math.sqrt(1 + 2.65 * S_squared)
            log_t_BH = a_hat + 1.51 * S_squared - term
            log_t_BB = a_hat + 1.51 * S_squared + term

            t_BH = 10 ** log_t_BH
            t_BB = 10 ** log_t_BB

            # Анализ готовности
            availability = self.calculate_availability(t_B)

            # Выводим результаты
            result = f"""РЕЗУЛЬТАТЫ РАСЧЕТА ДЛЯ МИГ-29:

Количество отказов (n): {n}
Среднее время восстановления t_B: {t_B:.2f} часов
Оценка параметра â: {a_hat:.3f}
Оценка параметра S: {S:.3f}

ДОВЕРИТЕЛЬНЫЕ ГРАНИЦЫ (β = {beta}):
Нижняя граница t_BH: {t_BH:.2f} часов
Верхняя граница t_BB: {t_BB:.2f} часов
Доверительный интервал: [{t_BH:.2f}; {t_BB:.2f}] часов

СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:
Минимальное время: {min(times):.2f} часов
Максимальное время: {max(times):.2f} часов
Стандартное отклонение: {np.std(times):.2f} часов

АНАЛИЗ ГОТОВНОСТИ:
{availability}
            """

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result)

            # Строим график
            self.plot_experimental_data(times, t_B, t_BH, t_BB)

        except Exception as e:
            messagebox.showerror("Ошибка расчета", f"Ошибка: {str(e)}")

    def calculate_availability(self, t_B):
        # Упрощенный анализ готовности (MTBF для МиГ-29 ~ 50-100 часов)
        mtbf = 80  # Среднее время между отказами (примерно)
        availability = mtbf / (mtbf + t_B) * 100

        if availability > 95:
            status = "ОТЛИЧНАЯ ГОТОВНОСТЬ"
        elif availability > 90:
            status = "ХОРОШАЯ ГОТОВНОСТЬ"
        elif availability > 85:
            status = "УДОВЛЕТВОРИТЕЛЬНАЯ ГОТОВНОСТЬ"
        else:
            status = "НИЗКАЯ ГОТОВНОСТЬ"

        return f"Коэффициент готовности: {availability:.1f}%\nСтатус: {status}"

    def plot_experimental_data(self, times, t_B, t_BH, t_BB):
        self.fig_experimental.clear()
        ax = self.fig_experimental.add_subplot(111)

        # Гистограмма
        n_bins = min(10, len(times) // 3)
        ax.hist(times, bins=n_bins, alpha=0.7, edgecolor='black', color='skyblue')

        # Вертикальные линии для среднего и доверительных границ
        ax.axvline(t_B, color='red', linestyle='-', linewidth=3, label=f'Среднее (t_B = {t_B:.2f} ч)')
        ax.axvline(t_BH, color='green', linestyle='--', linewidth=2, label=f'Нижняя граница ({t_BH:.2f} ч)')
        ax.axvline(t_BB, color='blue', linestyle='--', linewidth=2, label=f'Верхняя граница ({t_BB:.2f} ч)')

        ax.set_xlabel('Время восстановления (часы)')
        ax.set_ylabel('Частота отказов')
        ax.set_title('Распределение времени восстановления МиГ-29')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.canvas_experimental.draw()

    def calculate_design(self):
        try:
            # Собираем данные из таблицы
            elements = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                if len(values) == 4:
                    elements.append({
                        'type': values[0],
                        'count': float(values[1]),
                        'lambda': float(values[2]) * 1e-5,  # Переводим в 1/ч
                        'time': float(values[3])
                    })

            if not elements:
                messagebox.showerror("Ошибка", "Добавьте данные о конструктивных элементах")
                return

            # Вычисляем параметры по формулам 6-9
            omega_1 = 0
            numerator = 0

            for elem in elements:
                r_xi = elem['count']
                lambda_xi = elem['lambda']
                t_Bxi = elem['time']

                omega_1 += r_xi * lambda_xi
                numerator += r_xi * lambda_xi * t_Bxi

            # Среднее время восстановления (формула 6)
            t_B = numerator / omega_1 if omega_1 != 0 else 0

            # Анализ критических компонентов
            critical_components = []
            for elem in elements:
                contribution = elem['count'] * elem['lambda'] * elem['time'] / omega_1
                if contribution > 0.5:  # Компоненты с вкладом > 0.5 часа
                    critical_components.append((elem['type'], contribution))

            # Формируем результат
            result = f"""РЕЗУЛЬТАТЫ РАСЧЕТА ДЛЯ МИГ-29:

Параметр потока отказов ω: {omega_1:.6f} 1/ч
Среднее время восстановления t_B: {t_B:.2f} часов
Интенсивность отказов: {omega_1 * 1e5:.2f} × 10⁻⁵ 1/ч

ДЕТАЛИЗИРОВАННЫЙ РАСЧЕТ КОМПОНЕНТОВ:

"""
            total_contribution = 0
            for i, elem in enumerate(elements, 1):
                contribution = elem['count'] * elem['lambda'] * elem['time'] / omega_1
                total_contribution += contribution
                result += f"{i}. {elem['type']}: {contribution:.3f} ч ({contribution / t_B * 100:.1f}%)\n"

            result += f"\nОБЩАЯ ИНФОРМАЦИЯ:\n"
            result += f"Количество типов компонентов: {len(elements)}\n"
            result += f"Суммарный вклад: {total_contribution:.3f} ч\n"

            if critical_components:
                result += f"\nКРИТИЧЕСКИЕ КОМПОНЕНТЫ (вклад > 0.5 ч):\n"
                for comp, time in critical_components:
                    result += f"- {comp}: {time:.2f} ч\n"


            self.design_result_text.delete(1.0, tk.END)
            self.design_result_text.insert(1.0, result)

        except Exception as e:
            messagebox.showerror("Ошибка расчета", f"Ошибка: {str(e)}")


def main():
    root = tk.Tk()
    app = AircraftMaintenanceCalculator(root)
    root.mainloop()


if __name__ == "__main__":
    main()