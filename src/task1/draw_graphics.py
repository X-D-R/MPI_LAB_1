import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')

df['Эксперимент'] = df['Эксперимент'].astype(str)

# График для времени
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Количество точек'):
    plt.plot(grp['Количество процессов'], grp['Время'], label=f'{key} точек', marker='o')
plt.title('Время выполнения в зависимости от количества процессов и точек')
plt.xlabel('Количество процессов')
plt.ylabel('Время (сек.)')
plt.legend(title="Количество точек")
plt.grid(True)
plt.savefig('graphics/time_vs_processes.png')
plt.close()

# График для ускорения
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Количество точек'):
    plt.plot(grp['Количество процессов'], grp['Ускорение'], label=f'{key} точек', marker='o')
plt.title('Ускорение в зависимости от количества процессов и точек')
plt.xlabel('Количество процессов')
plt.ylabel('Ускорение')
plt.legend(title="Количество точек")
plt.grid(True)
plt.savefig('graphics/speedup_vs_processes.png')
plt.close()

# График для эффективности
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Количество точек'):
    plt.plot(grp['Количество процессов'], grp['Эффективность'], label=f'{key} точек', marker='o')
plt.title('Эффективность в зависимости от количества процессов и точек')
plt.xlabel('Количество процессов')
plt.ylabel('Эффективность')
plt.legend(title="Количество точек")
plt.grid(True)
plt.savefig('graphics/efficiency_vs_processes.png')
plt.close()

# График для погрешности
plt.figure(figsize=(10, 6))
for key, grp in df.groupby('Количество точек'):
    plt.plot(grp['Количество процессов'], grp['Погрешность'], label=f'{key} точек', marker='o')
plt.title('Погрешность в зависимости от количества процессов и точек')
plt.xlabel('Количество процессов')
plt.ylabel('Погрешность')
plt.legend(title="Количество точек")
plt.grid(True)
plt.savefig('graphics/error_vs_processes.png')
plt.close()
