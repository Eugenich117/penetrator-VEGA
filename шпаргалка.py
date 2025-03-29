    '''manager = multiprocessing.Manager()
    acceleration = manager.list([[] for _ in range(5)])
    napor = manager.list([manager.list() for _ in range(5)])
    TETTA = manager.list([manager.list() for _ in range(5)])
    X = manager.list([manager.list() for _ in range(5)])
    Y = manager.list([manager.list() for _ in range(5)])
    T = manager.list([manager.list() for _ in range(5)])
    PX = manager.list([manager.list() for _ in range(5)])
    nx = manager.list([manager.list() for _ in range(5)])
    V_MOD = manager.list([manager.list() for _ in range(5)])'''
    '''manager = multiprocessing.Manager()
    acceleration = manager.list()
    napor = manager.list()
    TETTA = manager.list()
    X = manager.list()
    Y = manager.list()
    T = manager.list()
    PX = manager.list()
    nx = manager.list()
    V_MOD = manager.list()'''
    '''iter = 30 #количество итераций
    dx = ['V', 'L', 'tetta', 'R', 'qk']
    equations = [dV_func, dL_func, dtetta_func, dR_func, qk_func]
    #with multiprocessing.Manager() as manager:

        # Создаем списки через менеджера

    acceleration = ([[] for _ in range(iter)])
    napor = ([[] for _ in range(iter)])
    TETTA = ([[] for _ in range(iter)])
    X = ([[] for _ in range(iter)])
    Y = ([[] for _ in range(iter)])
    T = ([[] for _ in range(iter)])
    PX = ([[] for _ in range(iter)])
    nx = ([[] for _ in range(iter)])
    V_MOD = ([[] for _ in range(iter)])
    V_WIND = ([[] for _ in range(iter)])
    WIND_ANGLE = ([[] for _ in range(iter)])
    Quantitiy_warm = ([[] for _ in range(iter)])
    Tomega = ([[] for _ in range(iter)])
    Qk = ([[] for _ in range(iter)])
    P_list = ([[] for _ in range(iter)])
    processes = []
    parent_conns = []
    child_conns = []

    tasks = []

    for i in range(iter):
        parent_conn, child_conn = multiprocessing.Pipe()  # Создаем пару для каждого процесса
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        tasks.append((i, equations, dx, child_conn))  # Формируем задачи для передачи в пул


    # Функция обратного вызова, которая будет вызываться по завершении каждой задачи

    # Создаем пул процессов
    pool = multiprocessing.Pool(processes=30)

    # Запускаем задачи асинхронно с отслеживанием завершения
    for task in tasks:
        pool.apply_async(compute_trajectory, task)'''

    '''for i in range(iter):
        parent_conn, child_conn = Pipe()  # Создаем пару для каждого процесса
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        p = multiprocessing.Process(target=compute_trajectory, args=(i, equations, dx, child_conn))
        p.start()
        processes.append(p)'''

    '''results = []
    while not queue.empty():
        results.append(queue.get())'''

    # Обработка результатов
'''    for i in range(iter):
        result = parent_conns[i].recv()
        (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
        local_acceleration, local_v_wind, local_wind_angle, local_Quantitiy_warm, local_Tomega, local_Qk, local_P) = result
        TETTA[i] = local_TETTA
        X[i] = local_X
        Y[i] = local_Y
        V_MOD[i] = local_V_MOD
        T[i] = local_T
        napor[i] = local_napor
        nx[i] = local_nx
        PX[i] = local_PX
        acceleration[i] = local_acceleration
        WIND_ANGLE[i] = local_wind_angle
        V_WIND[i] = local_v_wind
        Quantitiy_warm[i] = local_Quantitiy_warm
        Tomega[i] = local_Tomega
        Qk[i] = local_Qk
        P_list[i] = local_P

    data = {
        "acceleration": acceleration,
        "napor": napor,
        "TETTA": TETTA,
        "X": X,
        "Y": Y,
        "T": T,
        "PX": PX,
        "nx": nx,
        "V_MOD": [arr[-1:] for arr in V_MOD if arr],
    }

    # Перебираем каждый массив и находим min и max
    for key, values in data.items():
        all_values = np.concatenate(values)  # Объединяем все списки в один массив
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        print(f"{key}: min = {min_val}, max = {max_val}")'''
    with h5py.File("data.h5", "w") as h5f:
        # Создаем пул процессов
        with multiprocessing.Pool(processes=10) as pool:
            for task in tasks:
                pool.apply_async(compute_trajectory, task)

            # Читаем результаты и записываем в HDF5
            for i in range(iter_count):
                result = parent_conns[i].recv()
                (i, local_TETTA, local_X, local_Y, local_V_MOD, local_T, local_napor, local_nx, local_PX,
                 local_acceleration, local_v_wind, local_wind_angle, local_Quantitiy_warm, local_Tomega, local_Qk,
                 local_P) = result

                group = h5f.create_group(f"iter_{i}")

                # Записываем массивы
                group.create_dataset("TETTA", data=local_TETTA)
                group.create_dataset("X", data=local_X)
                group.create_dataset("Y", data=local_Y)
                group.create_dataset("V_MOD", data=local_V_MOD)
                group.create_dataset("T", data=local_T)
                group.create_dataset("napor", data=local_napor)
                group.create_dataset("nx", data=local_nx)
                group.create_dataset("PX", data=local_PX)
                group.create_dataset("acceleration", data=np.array(local_acceleration, dtype=np.float64))
                group.create_dataset("WIND_ANGLE", data=local_wind_angle)
                group.create_dataset("V_WIND", data=local_v_wind)
                group.create_dataset("Quantitiy_warm", data=local_Quantitiy_warm)
                group.create_dataset("Tomega", data=local_Tomega)
                group.create_dataset("Qk", data=local_Qk)
                group.create_dataset("P_list", data=local_P)

                print(f"✅ Данные для итерации {i} записаны в HDF5")

with h5py.File("data.h5", "r") as h5f:
    iter_2_data = h5f["iter_2"]["X"]
    chunk = iter_2_data[0:500]  # Загружаем только первые 500 элементов
    print(chunk)

with h5py.File("data.h5", "r") as h5f:
    iter_5_data = h5f["iter_5"]["V_MOD"][:]  # Загружаем массив скорости
    print("Длина массива V_MOD:", len(iter_5_data))