import sqlite3
def view_all_results():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM optimization_results')
    rows = cursor.fetchall()

    # Заголовки столбцов:
    column_names = [
        'id', 'best_last_V', 'best_last_P', 'best_p_soplar', 'best_tetta',
        'best_L', 'best_H', 'best_t', 'best_mass', 'best_chromosome',
        'population_size', 'generations', 'mutation_rate', 'elapsed_time', 'timestamp', 'PARAM_BOUNDS'
    ]

    for row in rows:
        print("=== Результат ===")
        for col_name, value in zip(column_names, row):
            print(f"{col_name}: {value}")
        print()

    conn.close()

view_all_results()