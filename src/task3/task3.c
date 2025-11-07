#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>

void initialize_matrix(double *matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = rand() % 10;
        }
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void matrix_multiply_sequential(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

double max_abs_diff(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static long count_lines(const char *path) {
    FILE *fr = fopen(path, "rb");
    if (!fr) return 0;
    long cnt = 0; int c;
    while ((c = fgetc(fr)) != EOF) if (c == '\n') cnt++;
    fclose(fr);
    return cnt;
}

static void append_result_csv(const char *path,
                                const char *algo_name,
                                int N,
                                int processes,
                                double time_seconds,
                                double speedup,
                                double efficiency,
                                double err)
{
    long lines = count_lines(path);
    FILE *f = fopen(path, "ab+");
    if (!f) return;

    if (lines == 0) {
        const char *hdr = "Эксперимент,Алгоритм,Размер N,Процессы,Время,Ускорение,Эффективность,Погрешность\n";
        fwrite(hdr, 1, strlen(hdr), f);
        lines = 1;
    }
    int experiment_id = (int)lines;

    fprintf(f, "%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
            experiment_id, algo_name, N, processes, time_seconds, speedup, efficiency, err);
    fclose(f);
}

int main(int argc, char *argv[]) {

    // mpicc -O3 task3.c -o task3 -lm

    // for p in 16; do     for N in 128 256 512 1024; do         echo "Запуск для алгоритма cannon, P=$p, N=$N";         mpirun --oversubscribe -np "$p" ./task3 "$N";     done; done

    int rank, size;
    int n = 512; // Размер матрицы по умолчанию
    int q;
    double start_time, end_time, parallel_time, sequential_time;
    const char *out_path = "results.csv";
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Проверка на квадратное количество процессов
    q = (int)sqrt(size);
    if (q * q != size) {
        if (rank == 0) {
            printf("Количество процессов должно быть полным квадратом!\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Чтение размера матрицы из аргументов
    if (argc > 1) n = atoi(argv[1]);
    if (n % q != 0) {
        if (rank == 0) {
            printf("Размер матрицы должен делиться на sqrt(количество процессов)!\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    int block_size = n / q;
    double *A_block = malloc(block_size * block_size * sizeof(double));
    double *B_block = malloc(block_size * block_size * sizeof(double));
    double *C_block = calloc(block_size * block_size, sizeof(double));
    double *A_temp = malloc(block_size * block_size * sizeof(double));
    double *B_temp = malloc(block_size * block_size * sizeof(double));
    
    // Создание декартовой топологии
    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    
    // Получение рангов соседей для сдвигов
    int left_rank, right_rank, up_rank, down_rank;
    MPI_Cart_shift(grid_comm, 1, -1, &left_rank, &right_rank); // Сдвиг по столбцам
    MPI_Cart_shift(grid_comm, 0, -1, &up_rank, &down_rank);   // Сдвиг по строкам
    
    // Инициализация матриц в процессе 0
    double *A = NULL, *B = NULL, *C_seq = NULL;
    if (rank == 0) {
        A = malloc(n * n * sizeof(double));
        B = malloc(n * n * sizeof(double));
        C_seq = malloc(n * n * sizeof(double));
        initialize_matrix(A, n, 1);
        initialize_matrix(B, n, 2);
        
        printf("Инициализация матриц завершена\n");
        
        // Последовательное умножение для проверки
        start_time = MPI_Wtime();
        matrix_multiply_sequential(A, B, C_seq, n);
        sequential_time = MPI_Wtime() - start_time;
        printf("Последовательное умножение завершено за %.6f сек\n", sequential_time);
    }
    
    // Рассылка блоков матриц - упрощенная версия
    if (rank == 0) {
        printf("Начало рассылки блоков...\n");
        // Процесс 0 копирует свои блоки
        for (int x = 0; x < block_size; x++) {
            memcpy(&A_block[x * block_size], 
                   &A[coords[0] * block_size * n + coords[1] * block_size + x * n],
                   block_size * sizeof(double));
            memcpy(&B_block[x * block_size], 
                   &B[coords[0] * block_size * n + coords[1] * block_size + x * n],
                   block_size * sizeof(double));
        }
        
        // Рассылка блоков другим процессам
        for (int dest = 1; dest < size; dest++) {
            int dest_coords[2];
            MPI_Cart_coords(grid_comm, dest, 2, dest_coords);
            
            // Отправка блока A
            for (int x = 0; x < block_size; x++) {
                double *src_ptr = &A[dest_coords[0] * block_size * n + dest_coords[1] * block_size + x * n];
                MPI_Send(src_ptr, block_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
            
            // Отправка блока B
            for (int x = 0; x < block_size; x++) {
                double *src_ptr = &B[dest_coords[0] * block_size * n + dest_coords[1] * block_size + x * n];
                MPI_Send(src_ptr, block_size, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        // Прием блоков другими процессами
        for (int x = 0; x < block_size; x++) {
            MPI_Recv(&A_block[x * block_size], block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int x = 0; x < block_size; x++) {
            MPI_Recv(&B_block[x * block_size], block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Рассылка блоков завершена\n");
    
    // Начальное смещение блоков согласно алгоритму Кэннона
    // Сдвиг матрицы A влево на coords[0] позиций
    for (int shift = 0; shift < coords[0]; shift++) {
        MPI_Sendrecv_replace(A_block, block_size * block_size, MPI_DOUBLE,
                           left_rank, 0, right_rank, 0, grid_comm, MPI_STATUS_IGNORE);
    }
    
    // Сдвиг матрицы B вверх на coords[1] позиций
    for (int shift = 0; shift < coords[1]; shift++) {
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE,
                           up_rank, 1, down_rank, 1, grid_comm, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Начальное смещение завершено\n");
    
    start_time = MPI_Wtime();
    
    // Основной цикл алгоритма Кэннона
    for (int step = 0; step < q; step++) {
        // Локальное умножение блоков
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    C_block[i * block_size + j] += 
                        A_block[i * block_size + k] * B_block[k * block_size + j];
                }
            }
        }
        
        // Циклический сдвиг блоков
        MPI_Sendrecv_replace(A_block, block_size * block_size, MPI_DOUBLE,
                           left_rank, 2, right_rank, 2, grid_comm, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE,
                           up_rank, 3, down_rank, 3, grid_comm, MPI_STATUS_IGNORE);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    parallel_time = MPI_Wtime() - start_time;
    
    if (rank == 0) printf("Параллельное умножение завершено\n");
    
    // Упрощенный сбор результатов
    double *C_par = NULL;
    if (rank == 0) {
        C_par = malloc(n * n * sizeof(double));
    }
    
    // Каждый процесс отправляет свой блок процессу 0
    if (rank != 0) {
        MPI_Send(C_block, block_size * block_size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    } else {
        // Процесс 0 копирует свой блок
        for (int x = 0; x < block_size; x++) {
            memcpy(&C_par[coords[0] * block_size * n + coords[1] * block_size + x * n],
                   &C_block[x * block_size], block_size * sizeof(double));
        }
        
        // Прием блоков от других процессов
        for (int src = 1; src < size; src++) {
            int src_coords[2];
            MPI_Cart_coords(grid_comm, src, 2, src_coords);
            
            double *recv_block = malloc(block_size * block_size * sizeof(double));
            MPI_Recv(recv_block, block_size * block_size, MPI_DOUBLE, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Копирование принятого блока в итоговую матрицу
            for (int x = 0; x < block_size; x++) {
                memcpy(&C_par[src_coords[0] * block_size * n + src_coords[1] * block_size + x * n],
                       &recv_block[x * block_size], block_size * sizeof(double));
            }
            free(recv_block);
        }
    }
    
    // Проверка результатов и вывод метрик
    if (rank == 0) {
        printf("Сбор результатов завершен\n");
        
        // Проверка корректности
        double err = 0.0;
        int errors = 0;
        for (int i = 0; i < n * n; i++) {
            if (fabs(C_par[i] - C_seq[i]) > 1e-5) {
                errors++;
                if (errors <= 5) { // Вывести только первые 5 ошибок
                    printf("Ошибка на позиции %d: ожидалось %.6f, получено %.6f\n", 
                           i, C_seq[i], C_par[i]);
                }
            }
        }
        
        err = max_abs_diff(C_par, C_seq, n * n);
        
        double speedup = (parallel_time > 0.0) ? (sequential_time / parallel_time) : 0.0;
        double efficiency = (size > 0) ? (speedup / size) : 0.0;
        
        printf("\n=== Результаты измерения производительности ===\n");
        printf("Алгоритм,Размер N,Процессы,Время,Ускорение,Эффективность,Погрешность\n");
        printf("cannon,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
            n, size, parallel_time, speedup, efficiency, err);
        
        // Запись в CSV файл
        append_result_csv(out_path, "cannon", n, size, parallel_time, speedup, efficiency, err);
        printf("\nРезультаты добавлены в файл: %s\n", out_path);
        
        free(A);
        free(B);
        free(C_seq);
        free(C_par);
    }
    
    free(A_block);
    free(B_block);
    free(C_block);
    free(A_temp);
    free(B_temp);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    
    return 0;
}