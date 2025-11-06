#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

double random_double(void);
double pi_sequential(long long num_points);
double pi_parallel(long long num_points, int rank, int size);
void append_result_csv(const char *path,
                    long long points,
                    int processes,
                    double time_seconds,
                    double speedup,
                    double efficiency,
                    double err);

int main(int argc, char** argv) {

    // mpicc task1.c -o task1
    // mpiexec -n 4 ./task1 10000000

    // for p in 2 4 8; do   for n in 1000000 5000000 10000000 50000000 100000000; do     echo "Run: P=$p, N=$n";     mpiexec -np "$p" ./task1 "$n";   done; done

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Использование: mpiexec -n <количество процессов> ./task1 <количество точек>\n");
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
    
    // Последовательная версия
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
    
    // Параллельная версия
    double parallel_time = 0.0, parallel_pi = 0.0;
    if (rank == 0) {
        printf("=== Параллельная версия ===\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    parallel_pi = pi_parallel(num_points, rank, size);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    MPI_Reduce(&local_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double speedup = 1.0, efficiency = 1.0;
    double err = 0.0;
    if (rank == 0) {
        err = fabs(parallel_pi - M_PI);
        printf("Результат: %.15f\n", parallel_pi);
        printf("Погрешность: %.15f\n", err);
        printf("Время выполнения: %.6f секунд\n", parallel_time);
        
        if (size > 1) {
            speedup = seq_time / parallel_time;
            efficiency = speedup / size;
            printf("Ускорение: %.6f\n", speedup);
            printf("Эффективность: %.6f\n", efficiency);
        }
        printf("\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    const char *out_path = "results.csv";
    
    if (rank == 0) {
        printf("=== Сводка результатов ===\n");
        printf("Количество точек,Количество процессов,Время,Ускорение,Эффективность,Погрешность\n");
        printf("%lld,%d,%.6f,%.6f,%.6f,%.6f\n", num_points, size, parallel_time, speedup, efficiency, err);

        append_result_csv(out_path, num_points, size, parallel_time, speedup, efficiency, err);
        printf("\nРезультаты добавлены в файл: %s\n", out_path);
    }
    
    MPI_Finalize();
    return 0;
}


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

static long count_lines(const char *path) {
    FILE *fr = fopen(path, "rb");
    if (!fr) return 0;
    long cnt = 0;
    int c;
    while ((c = fgetc(fr)) != EOF) {
        if (c == '\n') cnt++;
    }
    fclose(fr);
    return cnt;
}

void append_result_csv(const char *path,
                    long long points,
                    int processes,
                    double time_seconds,
                    double speedup,
                    double efficiency,
                    double err)
{
    FILE *f = fopen(path, "ab+");
    if (!f) return;

    long lines = count_lines(path);

    if (lines == 0) {
        const char *header =
            "Эксперимент,Количество точек,Количество процессов,Время,Ускорение,Эффективность,Погрешность\n";
        fwrite(header, 1, strlen(header), f);
        lines = 1;
    }

        int experiment_id = (int)lines;

    fprintf(f, "%d,%lld,%d,%.6f,%.6f,%.6f,%.6f\n",
            experiment_id, points, processes, time_seconds, speedup, efficiency, err);
            
    fclose(f);
}
