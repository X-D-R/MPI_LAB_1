import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cannon_results.csv')

df['Эксперимент'] = df['Эксперимент'].astype(str)

# График для времени
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Размер N'):
    for algo in grp['Алгоритм'].unique():
        grp_algo = grp[grp['Алгоритм'] == algo]
        plt.plot(grp_algo['Процессы'], grp_algo['Время'], label=f'{algo}, N={key}', marker='o')
plt.title('Время выполнения в зависимости от количества процессов, алгоритма и размера N')
plt.xlabel('Количество процессов')
plt.ylabel('Время (сек.)')
plt.legend(title="Алгоритм и Размер N")
plt.grid(True)
plt.savefig('graphics/time_vs_processes.png')
plt.close()

# График для ускорения
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Размер N'):
    for algo in grp['Алгоритм'].unique():
        grp_algo = grp[grp['Алгоритм'] == algo]
        plt.plot(grp_algo['Процессы'], grp_algo['Ускорение'], label=f'{algo}, N={key}', marker='o')
plt.title('Ускорение в зависимости от количества процессов, алгоритма и размера N')
plt.xlabel('Количество процессов')
plt.ylabel('Ускорение')
plt.legend(title="Алгоритм и Размер N")
plt.grid(True)
plt.savefig('graphics/speedup_vs_processes.png')
plt.close()

# График для эффективности
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Размер N'):
    for algo in grp['Алгоритм'].unique():
        grp_algo = grp[grp['Алгоритм'] == algo]
        plt.plot(grp_algo['Процессы'], grp_algo['Эффективность'], label=f'{algo}, N={key}', marker='o')
plt.title('Эффективность в зависимости от количества процессов, алгоритма и размера N')
plt.xlabel('Количество процессов')
plt.ylabel('Эффективность')
plt.legend(title="Алгоритм и Размер N")
plt.grid(True)
plt.savefig('graphics/efficiency_vs_processes.png')
plt.close()

# График для сравнения всех алгоритмов по времени для разных N
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Размер N'):
    for algo in grp['Алгоритм'].unique():
        grp_algo = grp[grp['Алгоритм'] == algo]
        plt.plot(grp_algo['Процессы'], grp_algo['Время'], label=f'{algo} для N={key}', marker='o')
plt.title('Сравнение времени выполнения для разных алгоритмов')
plt.xlabel('Количество процессов')
plt.ylabel('Время (сек.)')
plt.legend(title="Алгоритм и Размер N")
plt.grid(True)
plt.savefig('graphics/compare_algorithms_time.png')
plt.close()

# График для сравнения ускорения для разных N
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Размер N'):
    for algo in grp['Алгоритм'].unique():
        grp_algo = grp[grp['Алгоритм'] == algo]
        plt.plot(grp_algo['Процессы'], grp_algo['Ускорение'], label=f'{algo} для N={key}', marker='o')
plt.title('Сравнение ускорения для разных алгоритмов')
plt.xlabel('Количество процессов')
plt.ylabel('Ускорение')
plt.legend(title="Алгоритм и Размер N")
plt.grid(True)
plt.savefig('graphics/compare_algorithms_speedup.png')
plt.close()

# График для сравнения эффективности для разных N
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Размер N'):
    for algo in grp['Алгоритм'].unique():
        grp_algo = grp[grp['Алгоритм'] == algo]
        plt.plot(grp_algo['Процессы'], grp_algo['Эффективность'], label=f'{algo} для N={key}', marker='o')
plt.title('Сравнение эффективности для разных алгоритмов')
plt.xlabel('Количество процессов')
plt.ylabel('Эффективность')
plt.legend(title="Алгоритм и Размер N")
plt.grid(True)
plt.savefig('graphics/compare_algorithms_efficiency.png')
plt.close()
