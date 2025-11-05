#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

double random_double() {
    return 2 * (double)rand() / (double)RAND_MAX - 1;
}

double pi_sequential(long long num_points) {
    long long points_inside = 0;
    
    for (long long i = 0; i < num_points; i++) {
        double x = random_double();
        double y = random_double();
        
        if (x*x + y*y <= 1.0) {
            points_inside++;
        }
    }
    
    return 4.0 * (double)points_inside / (double)num_points;
}

double pi_parallel(long long num_points, int rank, int size) {
    long long local_points = num_points / size;
    long long local_inside = 0;
    
    unsigned int seed = time(NULL) + rank;
    
    for (long long i = 0; i < local_points; i++) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX * 2 - 1;
        double y = (double)rand_r(&seed) / (double)RAND_MAX * 2 - 1;
        
        if (x*x + y*y <= 1.0) {
            local_inside++;
        }
    }
    
    long long total_inside;
    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        return 4.0 * (double)total_inside / (double)num_points;
    }
    
    return 0.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Использование: mpiexec -n <количество процессов> <название файла> <количество_точек>\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    long long num_points = atoll(argv[1]);
    
    if (rank == 0) {
        printf("Вычисление числа pi\n");
        printf("Общее количество точек: %lld\n", num_points);
        printf("Количество процессов: %d\n", size);
        printf("Теоретическое значение pi: %.15f\n", M_PI);
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Тестирование последовательной версии (только процесс 0)
    double seq_time = 0.0, seq_pi = 0.0;
    if (rank == 0) {
        printf("=== Последовательная версия ===\n");
        double start_time = MPI_Wtime();
        seq_pi = pi_sequential(num_points);
        double end_time = MPI_Wtime();
        seq_time = end_time - start_time;
        
        printf("Результат: %.15f\n", seq_pi);
        printf("Погрешность: %.15f\n", fabs(seq_pi - M_PI));
        printf("Время выполнения: %.6f секунд\n", seq_time);
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Тестирование параллельной версии
    double parallel_time = 0.0, parallel_uniform_pi = 0.0;
    if (rank == 0) {
        printf("=== Параллельная версия (равномерное распределение) ===\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    parallel_uniform_pi = pi_parallel(num_points, rank, size);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    MPI_Reduce(&local_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Результат: %.15f\n", parallel_uniform_pi);
        printf("Погрешность: %.15f\n", fabs(parallel_uniform_pi - M_PI));
        printf("Время выполнения: %.6f секунд\n", parallel_time);
        
        if (size > 1) {
            double speedup = seq_time / parallel_time;
            double efficiency = speedup / size;
            printf("Ускорение: %.6f\n", speedup);
            printf("Эффективность: %.6f\n", efficiency);
        }
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Вывод результатов
    if (rank == 0) {
        printf("=== Сводка результатов ===\n");
        printf("Количество процессов,Время,Ускорение,Эффективность\n");
        printf("%d,%.6f", size, parallel_time);
        
        if (size > 1) {
            double speedup_uniform = seq_time / parallel_time;
            double efficiency_uniform = speedup_uniform / size;
            printf(",%.6f,%.6f", speedup_uniform, efficiency_uniform);
        }
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}