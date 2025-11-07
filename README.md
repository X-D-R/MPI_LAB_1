
# MPI

## Содержание
- [MPI](#mpi)
  - [Содержание](#содержание)
  - [Задание 1 — Монте‑Карло для π](#задание-1--монтекарло-для-π)
  - [Задание 2 — y = A·x: строки / столбцы / блоки](#задание-2--y--ax-строки--столбцы--блоки)
  - [Задание 3 — Матричное умножение: алгоритм Кэннона](#задание-3--матричное-умножение-алгоритм-кэннона)
  - [Результаты и выводы](#результаты-и-выводы)


## Задание 1 — Монте‑Карло для π
По методичке: из квадрата стороны 2 выбираем случайные точки и считаем долю попаданий в единичную окружность; отношение стремится к π/4. Реализована параллельная версия и построены графики времени, ускорения, эффективности и погрешности.

**Код**: [`src/task1/task1.c`](src/task1/task1.c)  
**CSV**: [`src/task1/results.csv`](src/task1/results.csv)

**Графики**:
- ![time_vs_processes](src/task1/graphics/time_vs_processes.png)
- ![speedup_vs_processes](src/task1/graphics/speedup_vs_processes.png)
- ![efficiency_vs_processes](src/task1/graphics/efficiency_vs_processes.png)
- ![error_vs_processes](src/task1/graphics/error_vs_processes.png)

## Задание 2 — y = A·x: строки / столбцы / блоки
Сравниваются три схемы распараллеливания умножения матрицы на вектор: разбиение по **строкам**, по **столбцам** и по **блокам**. Выполнены замеры для разных N и числа процессов p. Построены графики времени, ускорения, эффективности, а также **сравнительные** графики между алгоритмами.

**Код**: [`src/task2/task2.c`](src/task2/task2.c)  
**CSV**: [`src/task2/results.csv`](src/task2/results.csv)

**Графики по p**:
- ![time_vs_processes](src/task2/graphics/time_vs_processes.png)
- ![speedup_vs_processes](src/task2/graphics/speedup_vs_processes.png)
- ![efficiency_vs_processes](src/task2/graphics/efficiency_vs_processes.png)

**Сравнение алгоритмов**:
- ![compare_time](src/task2/graphics/compare_algorithms_time.png)
- ![compare_speedup](src/task2/graphics/compare_algorithms_speedup.png)
- ![compare_eff](src/task2/graphics/compare_algorithms_efficiency.png)

## Задание 3 — Матричное умножение: алгоритм Кэннона
Реализовано блочное матричное умножение по алгоритму **Кэннона**. Проведены замеры T_p и рассчитаны S_p, E_p в зависимости от размера задачи и числа процессов. Построены отдельные и сравнительные графики.

**Код**: [`src/task3/task3.c`](src/task3/task3.c)  
**CSV**: [`src/task3/cannon_results.csv`](src/task3/cannon_results.csv)

**Графики**:
- ![time_vs_processes](src/task3/graphics/time_vs_processes.png)
- ![speedup_vs_processes](src/task3/graphics/speedup_vs_processes.png)
- ![efficiency_vs_processes](src/task3/graphics/efficiency_vs_processes.png)
- ![compare_time](src/task3/graphics/compare_algorithms_time.png)
- ![compare_speedup](src/task3/graphics/compare_algorithms_speedup.png)
- ![compare_eff](src/task3/graphics/compare_algorithms_efficiency.png)


## Результаты и выводы

- **Задание 1 (π):** измерено 46 прогонов при p ∈ [2, 4, 8]. Эффективность падает с ростом p из‑за коммуникаций и уменьшения объёма работы на процесс, но ускорение положительное вплоть до 8 процессов. Погрешность снижается при увеличении числа точек.
- **Задание 2 (y=A·x):** наилучшее максимальное ускорение среди трёх подходов: row≈4.66, col≈4.86, block≈3.66. На больших N столбцы/строки масштабируются лучше; блочное разбиение стабильнее по памяти и коммуникациям.
- **Задание 3 (Кэннон):** ускорение растёт с p; при p=16 среднее по наборам размеров достигает ≈45.93. Эффективность ниже 1 на крупных p, что соответствует типичной картине из‑за обменов и латентностей.
