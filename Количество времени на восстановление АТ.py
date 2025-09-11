import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import json


class AircraftMaintenanceCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Калькулятор времени восстановления воздушных судов")
        self.root.geometry("1400x900")

        # Загружаем данные из файла или используем стандартные
        self.aircraft_data = self.load_data()

        # Словарь для хранения ссылок на деревья
        self.trees = {}

        self.create_widgets()

        # Сохраняем данные при закрытии программы
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_data(self):
        """Загружает данные из файла или возвращает стандартные данные"""
        data_file = "aircraft_data.json"

        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    print("Данные загружены из файла")
                    return saved_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Ошибка загрузки файла: {e}, используем стандартные данные")
                return self.get_default_data()
        else:
            print("Файл не найден, используем стандартные данные")
            return self.get_default_data()

    def get_default_data(self):
        """Возвращает стандартные данные"""
        return {
            "МиГ-29": self.get_mig29_data(),
            "Су-27": self.get_su27_data(),
            "Су-35": self.get_su35_data(),
            "Ту-160": self.get_tu160_data(),
            "Ил-76": self.get_il76_data(),
            "Ан-124": self.get_an124_data(),
            "SSJ-100": self.get_ssj100_data(),
            "МС-21": self.get_ms21_data()
        }

    def save_data(self):
        """Сохраняет данные в файл"""
        data_file = "aircraft_data.json"

        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(self.aircraft_data, f, ensure_ascii=False, indent=2)
            print("Данные успешно сохранены в файл")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить данные: {str(e)}")

    def on_closing(self):
        """Обработчик закрытия окна"""
        self.save_data()
        self.root.destroy()

    def get_mig29_data(self):
        return {
            "experimental_times": "12.5, 15.0, 18.2, 10.8, 22.5, 16.8, 13.2, 24.0, 19.5, 16.2, 15.8, 21.2, 17.5, 20.8, 14.0, 26.5, 18.8, 20.0, 15.5, 23.8, 16.2, 19.5, 15.8, 25.2, 17.5",
            "components": [
                ("Двигатель РД-33", "2", "1.6", "28.0"),
                ("Система управления", "1", "1.2", "18.2"),
                ("Радиолокационная станция", "1", "2.0", "22.5"),
                ("Система навигации", "1", "0.9", "15.5"),
                ("Гидравлическая система", "1", "1.4", "20.8"),
                ("Топливная система", "1", "1.1", "14.2"),
                ("Электрооборудование", "1", "1.5", "19.5"),
                ("Шасси", "3", "0.6", "22.2")
            ],
            "description": "Многоцелевой истребитель 4-го поколения"
        }

    def get_su27_data(self):
        return {
            "experimental_times": "16.2, 22.5, 18.8, 26.2, 21.8, 23.2, 17.5, 31.2, 25.8, 20.5, 19.0, 28.5, 22.0, 27.0, 18.2, 34.8, 23.2, 24.5, 19.0, 32.0, 21.8, 25.8, 18.2, 33.5, 22.0",
            "components": [
                ("Двигатель АЛ-31Ф", "2", "1.4", "32.5"),
                ("Система управления", "1", "1.1", "21.5"),
                ("Радиолокационная станция", "1", "1.8", "27.0"),
                ("Система навигации", "1", "0.8", "16.8"),
                ("Гидравлическая система", "1", "1.3", "24.0"),
                ("Топливная система", "1", "1.0", "17.5"),
                ("Электрооборудование", "1", "1.4", "22.8"),
                ("Шасси", "3", "0.5", "25.5")
            ],
            "description": "Многоцелевой истребитель 4-го поколения"
        }

    def get_su35_data(self):
        return {
            "experimental_times": "18.8, 22.5, 24.0, 16.2, 28.0, 21.5, 15.0, 30.5, 24.2, 19.0, 17.5, 26.8, 20.2, 24.5, 18.8, 32.0, 22.8, 24.2, 18.8, 28.2, 21.5, 24.0, 19.0, 31.8, 22.8",
            "components": [
                ("Двигатель АЛ-41Ф-1С", "2", "1.2", "30.8"),
                ("Цифровая СУ", "1", "1.0", "19.0"),
                ("РЛС с АФАР", "1", "1.6", "25.2"),
                ("ИНС", "1", "0.7", "15.2"),
                ("Гидравлическая система", "1", "1.1", "22.5"),
                ("Топливная система", "1", "0.9", "14.0"),
                ("Электрооборудование", "1", "1.2", "20.2"),
                ("Шасси", "3", "0.4", "23.0")
            ],
            "description": "Многоцелевой истребитель 4++ поколения"
        }

    def get_tu160_data(self):
        return {
            "experimental_times": "42.5, 56.2, 38.8, 65.5, 48.2, 55.8, 40.5, 82.2, 63.5, 46.8, 52.2, 68.5, 49.5, 60.8, 44.2, 78.5, 56.2, 63.5, 48.0, 72.2, 54.5, 62.0, 49.5, 81.2, 58.8",
            "components": [
                ("Двигатель НК-32", "4", "2.5", "58.0"),
                ("Система управления", "1", "1.8", "45.5"),
                ("Навигационный комплекс", "1", "1.5", "38.2"),
                ("Топливная система", "1", "2.0", "52.0"),
                ("Гидравлическая система", "1", "1.6", "42.5"),
                ("Электрооборудование", "1", "2.2", "60.5"),
                ("Система вооружения", "1", "1.9", "48.8"),
                ("Шасси", "12", "1.2", "38.5")
            ],
            "description": "Стратегический бомбардировщик-ракетоносец"
        }

    def get_il76_data(self):
        return {
            "experimental_times": "35.2, 45.5, 32.0, 52.2, 40.0, 46.8, 34.5, 62.5, 48.2, 38.8, 42.5, 58.0, 43.8, 50.5, 36.5, 66.2, 45.5, 52.0, 40.2, 59.5, 44.0, 50.2, 42.5, 65.8, 48.8",
            "components": [
                ("Двигатель ПС-90", "4", "1.8", "52.5"),
                ("Система управления", "1", "1.4", "40.8"),
                ("Навигационный комплекс", "1", "1.2", "33.2"),
                ("Топливная система", "1", "1.6", "45.2"),
                ("Гидравлическая система", "1", "1.3", "38.0"),
                ("Электрооборудование", "1", "1.7", "46.0"),
                ("Грузовая система", "1", "1.5", "40.5"),
                ("Шасси", "10", "1.0", "36.8")
            ],
            "description": "Военно-транспортный самолет"
        }

    def get_an124_data(self):
        return {
            "experimental_times": "52.5, 68.2, 48.0, 78.8, 60.5, 66.8, 54.2, 92.5, 72.2, 58.8, 65.5, 82.0, 64.8, 70.5, 58.2, 88.2, 68.0, 75.5, 62.2, 80.5, 66.8, 72.0, 65.5, 92.2, 70.8",
            "components": [
                ("Двигатель Д-18Т", "4", "2.2", "80.5"),
                ("Система управления", "1", "1.7", "62.8"),
                ("Навигационный комплекс", "1", "1.4", "50.2"),
                ("Топливная система", "1", "1.9", "68.5"),
                ("Гидравлическая система", "1", "1.6", "58.5"),
                ("Электрооборудование", "1", "2.0", "72.2"),
                ("Грузовая система", "1", "1.8", "62.5"),
                ("Шасси", "16", "1.3", "52.0")
            ],
            "description": "Тяжелый транспортный самолет"
        }

    def get_ssj100_data(self):
        return {
            "experimental_times": "25.2, 32.8, 22.5, 40.2, 30.8, 36.0, 25.8, 48.5, 38.2, 28.5, 33.2, 45.5, 32.0, 40.5, 26.2, 52.0, 34.8, 42.8, 28.5, 46.0, 33.5, 38.2, 30.8, 50.8, 36.0",
            "components": [
                ("Двигатель SaM146", "2", "1.0", "38.2"),
                ("Цифровая СУ", "1", "0.8", "26.0"),
                ("Авионика", "1", "1.2", "35.5"),
                ("Навигационный комплекс", "1", "0.7", "22.2"),
                ("Топливная система", "1", "0.9", "29.5"),
                ("Гидравлическая система", "1", "0.8", "27.2"),
                ("Электрооборудование", "1", "1.1", "32.0"),
                ("Шасси", "6", "0.6", "30.8")
            ],
            "description": "Региональный пассажирский самолет"
        }

    def get_ms21_data(self):
        return {
            "experimental_times": "22.8, 30.2, 20.0, 38.5, 27.2, 32.5, 22.5, 45.8, 34.8, 26.0, 30.8, 42.8, 28.5, 35.8, 24.8, 48.2, 32.2, 38.2, 27.2, 43.5, 31.0, 35.8, 28.5, 46.0, 33.5",
            "components": [
                ("Двигатель ПД-14", "2", "0.9", "34.5"),
                ("Цифровая СУ", "1", "0.7", "23.5"),
                ("Авионика", "1", "1.0", "31.8"),
                ("Навигационный комплекс", "1", "0.6", "19.8"),
                ("Топливная система", "1", "0.8", "26.0"),
                ("Гидравлическая система", "1", "0.7", "24.8"),
                ("Электрооборудование", "1", "0.9", "29.5"),
                ("Шасси", "8", "0.5", "27.2"),
                ("Композитные панели", "1", "0.4", "20.0")
            ],
            "description": "Магистральный пассажирский самолет"
        }

    def create_widgets(self):
        # Создаем notebook для разделения на вкладки по типам самолетов
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Создаем вкладки для каждого типа самолета
        for aircraft_type in self.aircraft_data.keys():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=aircraft_type)
            self.create_aircraft_tab(frame, aircraft_type)

        # Вкладка с сравнением
        frame_comparison = ttk.Frame(notebook)
        notebook.add(frame_comparison, text="Сравнение самолетов")
        self.create_comparison_tab(frame_comparison)

        # Вкладка со справочной информацией
        frame_reference = ttk.Frame(notebook)
        notebook.add(frame_reference, text="Справочник")
        self.create_reference_tab(frame_reference)

        # Кнопка для ручного сохранения в главном окне
        save_button = ttk.Button(self.root, text="Сохранить данные", command=self.save_data)
        save_button.pack(pady=5)

    def create_aircraft_tab(self, parent, aircraft_type):
        # Заголовок с описанием самолета
        desc_frame = ttk.Frame(parent)
        desc_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(desc_frame, text=f"{aircraft_type} - {self.aircraft_data[aircraft_type]['description']}",
                  font=('Arial', 12, 'bold')).pack(pady=5)

        # Ввод данных о времени восстановления
        input_frame = ttk.LabelFrame(parent, text=f"Экспериментальные данные для {aircraft_type} (часы)")
        input_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(input_frame, text="Введите время восстановления через запятую:").pack(pady=5)
        time_entry = ttk.Entry(input_frame, width=100)
        time_entry.pack(pady=5)
        time_entry.insert(0, self.aircraft_data[aircraft_type]['experimental_times'])

        # Уровень доверия
        ttk.Label(input_frame, text="Доверительная вероятность β:").pack(pady=5)
        confidence_var = tk.StringVar(value="0.95")
        confidence_combo = ttk.Combobox(input_frame, textvariable=confidence_var,
                                        values=["0.80", "0.90", "0.95", "0.975", "0.990", "0.995", "0.9975", "0.999"])
        confidence_combo.pack(pady=5)

        # Кнопка расчета
        ttk.Button(input_frame, text="Рассчитать",
                   command=lambda: self.calculate_experimental(time_entry.get(), confidence_var.get(),
                                                               result_text, fig, canvas, aircraft_type)).pack(pady=10)

        # Таблица компонентов
        table_frame = ttk.LabelFrame(parent, text=f"Конструктивные элементы {aircraft_type}")
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)

        columns = ("Компонент", "Количество", "Интенсивность отказов (10⁻⁵/ч)", "Время восстановления (ч)")
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Сохраняем ссылку на дерево
        self.trees[aircraft_type] = tree

        # Заполняем таблицу данными из aircraft_data
        for component in self.aircraft_data[aircraft_type]['components']:
            tree.insert("", tk.END, values=component)

        # Добавляем возможность редактирования двойным кликом
        tree.bind("<Double-1>", lambda event: self.on_double_click(event, tree, aircraft_type))

        # Кнопка расчета проектирования
        ttk.Button(parent, text="Рассчитать время восстановления",
                   command=lambda: self.calculate_design(tree, design_result_text)).pack(pady=10)

        # Результаты
        result_frame = ttk.Frame(parent)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Левая часть - экспериментальные результаты
        exp_frame = ttk.LabelFrame(result_frame, text="Результаты экспериментального расчета")
        exp_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5)

        result_text = scrolledtext.ScrolledText(exp_frame, height=12)
        result_text.pack(fill='both', expand=True, padx=5, pady=5)

        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvasTkAgg(fig, exp_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Правая часть - результаты проектирования
        design_frame = ttk.LabelFrame(result_frame, text="Результаты проектирования")
        design_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=5)

        design_result_text = scrolledtext.ScrolledText(design_frame, height=20)
        design_result_text.pack(fill='both', expand=True, padx=5, pady=5)

    def on_double_click(self, event, tree, aircraft_type):
        """Обработчик двойного клика для редактирования ячейки"""
        region = tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        column = tree.identify_column(event.x)
        column_index = int(column[1:]) - 1
        item = tree.identify_row(event.y)

        # Разрешаем редактирование только колонки "Время восстановления (ч)" (индекс 3)
        if column_index != 3:
            return

        # Получаем текущее значение
        current_values = tree.item(item, "values")
        current_value = current_values[column_index]

        # Создаем окно редактирования
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Редактирование времени восстановления")
        edit_window.geometry("300x100")
        edit_window.transient(self.root)
        edit_window.grab_set()

        ttk.Label(edit_window, text="Новое значение времени восстановления (ч):").pack(pady=5)

        edit_var = tk.StringVar(value=current_value)
        edit_entry = ttk.Entry(edit_window, textvariable=edit_var, width=20)
        edit_entry.pack(pady=5)
        edit_entry.select_range(0, tk.END)
        edit_entry.focus()

        def save_edit():
            new_value = edit_var.get()
            try:
                # Проверяем, что введено число
                float_value = float(new_value)
                # Обновляем значение в таблице
                new_values = list(current_values)
                new_values[column_index] = str(float_value)
                tree.item(item, values=new_values)
                # Обновляем данные в aircraft_data
                self.update_aircraft_data(aircraft_type, tree, item, new_values)
                # Автоматически сохраняем изменения
                self.save_data()
                edit_window.destroy()
            except ValueError:
                messagebox.showerror("Ошибка", "Введите числовое значение")

        ttk.Button(edit_window, text="Сохранить", command=save_edit).pack(pady=5)
        edit_window.bind('<Return>', lambda e: save_edit())

    def update_aircraft_data(self, aircraft_type, tree, item, new_values):
        """Обновляем данные в словаре aircraft_data"""
        # Получаем индекс элемента в таблице
        item_index = tree.index(item)

        # Обновляем соответствующий компонент
        if item_index < len(self.aircraft_data[aircraft_type]['components']):
            # Преобразуем обратно в tuple для сохранения структуры данных
            self.aircraft_data[aircraft_type]['components'][item_index] = tuple(new_values)

    def create_comparison_tab(self, parent):
        comparison_frame = ttk.LabelFrame(parent, text="Сравнение времени восстановления самолетов")
        comparison_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Кнопка для выполнения сравнения
        ttk.Button(comparison_frame, text="Выполнить сравнение",
                   command=lambda: self.perform_comparison(comparison_text, fig_comp, canvas_comp)).pack(pady=10)

        # Текстовое поле для результатов сравнения
        comparison_text = scrolledtext.ScrolledText(comparison_frame, height=15)
        comparison_text.pack(fill='both', expand=True, padx=10, pady=5)

        # График сравнения
        fig_comp = Figure(figsize=(10, 6))
        canvas_comp = FigureCanvasTkAgg(fig_comp, comparison_frame)
        canvas_comp.get_tk_widget().pack(fill='both', expand=True)

    def create_reference_tab(self, parent):
        info_text = """
        СПРАВОЧНАЯ ИНФОРМАЦИЯ ПО САМОЛЕТАМ:

        МИГ-29: Многоцелевой истребитель 4-го поколения
        СУ-27: Многоцелевой истребитель 4-го поколения
        СУ-35: Многоцелевой истребитель 4++ поколения
        ТУ-160: Стратегический бомбардировщик-ракетоносец
        ИЛ-76: Военно-транспортный самолет
        АН-124: Тяжелый транспортный самолет
        SSJ-100: Региональный пассажирский самолет
        МС-21: Магистральный пассажирский самолет

        ТИПОВЫЕ ЗНАЧЕНИЯ ВРЕМЕНИ ВОССТАНОВЛЕНИЯ:
        - Истребители: 3.5-6.5 часов
        - Бомбардировщики: 8.0-18.0 часов
        - Транспортные: 6.0-14.0 часов
        - Пассажирские: 4.0-9.0 часов

        ФОРМУЛЫ РАСЧЕТА:
        t_B = (1/n) * Σ τ_B_i
        Доверительные границы рассчитываются по логарифмически нормальному распределению

        КВАНИТИЛИ НОРМАЛЬНОО РАСПРЕДЕЛЕНИЯ:
        β: 0.80, 0.90, 0.95, 0.975, 0.990, 0.995, 0.9975, 0.999
        u_γ: 0.842, 1.282, 1.645, 1.960, 2.326, 2.576, 2.807, 3.090
        """

        text_widget = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        text_widget.insert(1.0, info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)

    def calculate_experimental(self, times_str, confidence_str, result_text, fig, canvas, aircraft_type):
        try:
            times = [float(x.strip()) for x in times_str.split(',') if x.strip()]

            if not times:
                messagebox.showerror("Ошибка", "Введите данные о времени восстановления")
                return

            n = len(times)
            t_B = sum(times) / n

            log_times = [math.log10(t) for t in times]
            a_hat = sum(log_times) / n

            S_squared = sum((x - a_hat) ** 2 for x in log_times) / (n - 1)
            S = math.sqrt(S_squared)

            beta = float(confidence_str)
            u_gamma_dict = {
                0.80: 0.842, 0.90: 1.282, 0.95: 1.645, 0.975: 1.960,
                0.990: 2.326, 0.995: 2.576, 0.9975: 2.807, 0.999: 3.090
            }
            u_gamma = u_gamma_dict.get(beta, 1.645)

            term = u_gamma * (S / math.sqrt(n)) * math.sqrt(1 + 2.65 * S_squared)
            log_t_BH = a_hat + 1.51 * S_squared - term
            log_t_BB = a_hat + 1.51 * S_squared + term

            t_BH = 10 ** log_t_BH
            t_BB = 10 ** log_t_BB

            availability = self.calculate_availability(t_B, aircraft_type)

            result = f"""РЕЗУЛЬТАТЫ РАСЧЕТА ДЛЯ {aircraft_type}:

Количество отказов: {n}
Среднее время восстановления: {t_B:.2f} часов
Доверительный интервал ({beta}): [{t_BH:.2f}; {t_BB:.2f}] часов

СТАТИСТИКА:
Минимум: {min(times):.2f} ч, Максимум: {max(times):.2f} ч
Стандартное отклонение: {np.std(times):.2f} ч

АНАЛИЗ ГОТОВНОСТИ:
{availability}
            """

            result_text.delete(1.0, tk.END)
            result_text.insert(1.0, result)

            self.plot_data(times, t_B, t_BH, t_BB, fig, canvas, aircraft_type)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка расчета: {str(e)}")

    def calculate_availability(self, t_B, aircraft_type):
        mtbf_values = {
            "МиГ-29": 80, "Су-27": 85, "Су-35": 90,
            "Ту-160": 120, "Ил-76": 150, "Ан-124": 140,
            "SSJ-100": 200, "МС-21": 220
        }

        mtbf = mtbf_values.get(aircraft_type, 100)
        availability = mtbf / (mtbf + t_B) * 100

        if availability > 95:
            status = "ОТЛИЧНАЯ"
        elif availability > 90:
            status = "ХОРОШАЯ"
        elif availability > 85:
            status = "УДОВЛЕТВОРИТЕЛЬНАЯ"
        else:
            status = "НИЗКАЯ"

        return f"Коэффициент готовности: {availability:.1f}%\nСтатус: {status}"

    def plot_data(self, times, t_B, t_BH, t_BB, fig, canvas, aircraft_type):
        fig.clear()
        ax = fig.add_subplot(111)

        n_bins = min(10, len(times) // 3)
        ax.hist(times, bins=n_bins, alpha=0.7, edgecolor='black', color='lightblue')

        ax.axvline(t_B, color='red', linewidth=3, label=f'Среднее: {t_B:.2f} ч')
        ax.axvline(t_BH, color='green', linestyle='--', label=f'Нижняя: {t_BH:.2f} ч')
        ax.axvline(t_BB, color='blue', linestyle='--', label=f'Верхняя: {t_BB:.2f} ч')

        ax.set_xlabel('Время восстановления (часы)')
        ax.set_ylabel('Частота')
        ax.set_title(f'Распределение времени восстановления - {aircraft_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas.draw()

    def calculate_design(self, tree, result_text):
        try:
            elements = []
            for item in tree.get_children():
                values = tree.item(item)['values']
                if len(values) == 4:
                    elements.append({
                        'count': float(values[1]),
                        'lambda': float(values[2]) * 1e-5,
                        'time': float(values[3])
                    })

            omega_1 = sum(elem['count'] * elem['lambda'] for elem in elements)
            numerator = sum(elem['count'] * elem['lambda'] * elem['time'] for elem in elements)
            t_B = numerator / omega_1 if omega_1 != 0 else 0

            result = f"""РЕЗУЛЬТАТЫ ПРОЕКТИРОВАНИЯ:

Параметр потока отказов: {omega_1:.6f} 1/ч
Среднее время восстановления: {t_B:.2f} часов
Интенсивность отказов: {omega_1 * 1e5:.2f} × 10⁻⁵ 1/ч

ДЕТАЛИЗАЦИЯ ПО КОМПОНЕНТАМ:
"""
            for item in tree.get_children():
                values = tree.item(item)['values']
                if len(values) == 4:
                    count = float(values[1])
                    lambda_val = float(values[2]) * 1e-5
                    time_val = float(values[3])
                    contribution = count * lambda_val * time_val / omega_1
                    result += f"- {values[0]}: {contribution:.3f} ч ({contribution / t_B * 100:.1f}%)\n"

            result_text.delete(1.0, tk.END)
            result_text.insert(1.0, result)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка расчета: {str(e)}")

    def perform_comparison(self, comparison_text, fig, canvas):
        try:
            results = {}

            for aircraft in self.aircraft_data.keys():
                # Имитируем расчет для каждого самолета
                times = [float(x.strip()) for x in self.aircraft_data[aircraft]['experimental_times'].split(',') if
                         x.strip()]
                if times:
                    t_B = sum(times) / len(times)
                    results[aircraft] = t_B

            # Сортируем результаты
            sorted_results = sorted(results.items(), key=lambda x: x[1])

            # Формируем текст сравнения
            comparison_result = "СРАВНЕНИЕ СРЕДНЕГО ВРЕМЕНИ ВОССТАНОВЛЕНИЯ:\n\n"
            for i, (aircraft, time) in enumerate(sorted_results, 1):
                comparison_result += f"{i}. {aircraft}: {time:.2f} часов\n"

            comparison_text.delete(1.0, tk.END)
            comparison_text.insert(1.0, comparison_result)

            # Строим график сравнения
            self.plot_comparison(results, fig, canvas)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сравнения: {str(e)}")

    def plot_comparison(self, results, fig, canvas):
        fig.clear()
        ax = fig.add_subplot(111)

        aircrafts = list(results.keys())
        times = list(results.values())

        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon',
                  'lightseagreen', 'lightpink', 'lightyellow', 'lightgray']

        bars = ax.bar(aircrafts, times, color=colors[:len(aircrafts)])
        ax.set_ylabel('Время восстановления (часы)')
        ax.set_title('Сравнение времени восстановления самолетов')
        ax.tick_params(axis='x', rotation=45)

        # Добавляем значения на столбцы
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{time:.1f}', ha='center', va='bottom')

        fig.tight_layout()
        canvas.draw()


def main():
    root = tk.Tk()
    app = AircraftMaintenanceCalculator(root)
    root.mainloop()


if __name__ == "__main__":
    main()