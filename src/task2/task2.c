#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

static void fill_random_matrix(double *A, int N);
static void fill_random_vector(double *x, int N);
static void matvec_seq(const double *A, const double *x, double *y, int N);
static double max_abs_diff(const double *a, const double *b, int n);

static long count_lines(const char *path);
static void append_result_csv(const char *path,
                                    const char *algo_name,
                                    int N,
                                    int processes,
                                    double time_seconds,
                                    double speedup,
                                    double efficiency,
                                    double err);

static void matvec_row_mpi(const double *A_root, const double *x_root, double *y_root, int N,
                           int rank, int size, double *t_out);

static void matvec_col_mpi(const double *A_root, const double *x_root, double *y_root, int N,
                           int rank, int size, double *t_out);

static void matvec_block_mpi(const double *A_root, const double *x_root, double *y_root, int N,
                             int rank, int size, double *t_out);


int main(int argc, char **argv) {

    // mpicc -O3 task2.c -o task2
    //   mpirun -np 4 ./task2 4096 row
    //   mpirun -np 4 ./task2 4096 col
    //   mpirun -np 4 ./task2 4096 block

    /*
    for algo in row col; do
        for p in 2 4 8; do
            for N in 1024 2048 4096 8192; do
                echo "Запуск для алгоритма $algo, P=$p, N=$N"
                mpiexec -np "$p" ./task2 "$N" "$algo"
                done
            done
        done
    
    for p in 4 9; do
        for N in 1024 2048 4096 8192; do
            echo "Запуск для алгоритма block, P=$p, N=$N"
            mpiexec -np "$p" ./task2 "$N" block
            done
        done
    */

    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            printf("Использование:\n");
            printf("  mpirun -np <процессы> ./task2 <N> <row|col|block> \n");
        }
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const char *algo = argv[2];
    const char *out_path = "results.csv";

    if (rank == 0) {
        printf("Матрица-Вектор (N=%d), алгоритм=%s, процессы=%d\n", N, algo, size);
    }

    // Данные только на root (rank 0)
    double *A = NULL, *x = NULL, *y = NULL, *y_ref = NULL;

    if (rank == 0) {
        A = (double*)malloc((size_t)N * N * sizeof(double));
        x = (double*)malloc((size_t)N * sizeof(double));
        y = (double*)malloc((size_t)N * sizeof(double));
        y_ref = (double*)malloc((size_t)N * sizeof(double));
        if (!A || !x || !y || !y_ref) {
            fprintf(stderr, "Allocation failed on root\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        fill_random_matrix(A, N);
        fill_random_vector(x, N);
    }

    // Последовательный запуск
    double time_seq = 0.0;
    if (rank == 0) {
        double t0 = MPI_Wtime();
        matvec_seq(A, x, y_ref, N);
        double t1 = MPI_Wtime();
        time_seq = t1 - t0;
        printf("Seq time: %.6f s\n", time_seq);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Параллельный запуск выбранного алгоритма
    double time_par = 0.0;
    if (strcmp(algo, "row") == 0) {
        matvec_row_mpi(A, x, y, N, rank, size, &time_par);
    } else if (strcmp(algo, "col") == 0) {
        matvec_col_mpi(A, x, y, N, rank, size, &time_par);
    } else if (strcmp(algo, "block") == 0) {
        matvec_block_mpi(A, x, y, N, rank, size, &time_par);
    } else {
        if (rank == 0) fprintf(stderr, "Неизвестный алгоритм: %s\n", algo);
        if (A) free(A);
        if (x) free(x); 
        if (y) free(y); 
        if (y_ref) free(y_ref);
        MPI_Finalize();
        return 2;
    }

    // Проверка корректности
    double err = 0.0;
    if (rank == 0) {
        err = max_abs_diff(y, y_ref, N);
        printf("Max |y - y_ref| = %.6e\n", err);

        double speedup = (time_par > 0.0) ? (time_seq / time_par) : 0.0;
        double efficiency = (size > 0) ? (speedup / size) : 0.0;

        printf("\n=== Сводка результатов ===\n");
        printf("Алгоритм,Размер N,Процессы,Время,Ускорение,Эффективность,Погрешность\n");

        append_result_csv(out_path, algo, N, size, time_par, speedup, efficiency, err);
        printf("%s,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
            algo, N, size, time_par, speedup, efficiency, err);
        printf("\nРезультаты добавлены в файл: %s\n", out_path);
    }

    if (A) free(A);
    if (x) free(x);
    if (y) free(y);
    if (y_ref) free(y_ref);

    MPI_Finalize();
    return 0;
}

static void fill_random_matrix(double *A, int N) {
    for (int i = 0; i < N*N; ++i) {
        A[i] = (double)(rand() % 10); // 0..9
    }
}

static void fill_random_vector(double *x, int N) {
    for (int i = 0; i < N; ++i) {
        x[i] = (double)(rand() % 10); // 0..9
    }
}

static void matvec_seq(const double *A, const double *x, double *y, int N) {
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        const double *Ai = A + (size_t)i * N;
        for (int j = 0; j < N; ++j) s += Ai[j] * x[j];
        y[i] = s;
    }
}

static double max_abs_diff(const double *a, const double *b, int n) {
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
                                double err
                                )
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

static void matvec_row_mpi(const double *A_root, const double *x_root, double *y_root, int N,
                           int rank, int size, double *t_out)
{
    int base = N / size, rem = N % size;
    int my_rows = base + (rank < rem ? 1 : 0);

    // countsA/displsA в ЭЛЕМЕНТАХ МАТРИЦЫ
    int *countsA = NULL, *displsA = NULL;
    // countsY/displsY в КОЛИЧЕСТВЕ СТРОК
    int *countsY = NULL, *displsY = NULL;

    if (rank == 0) {
        countsA = (int*)malloc(size * sizeof(int));
        displsA = (int*)malloc(size * sizeof(int));
        countsY = (int*)malloc(size * sizeof(int));
        displsY = (int*)malloc(size * sizeof(int));

        int off_rows = 0;
        for (int r = 0; r < size; ++r) {
            int rows_r = base + (r < rem ? 1 : 0);
            countsA[r] = rows_r * N;  // элементы матрицы
            displsA[r] = off_rows * N;  // смещение в элементах
            countsY[r] = rows_r;  // строки результата
            displsY[r] = off_rows;  // смещение в строках
            off_rows += rows_r;
        }
    }

    // x -> всем
    double *x = (double*)malloc((size_t)N * sizeof(double));
    if (rank == 0) memcpy(x, x_root, (size_t)N * sizeof(double));
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // A -> локально
    double *A_local = (double*)malloc((size_t)my_rows * N * sizeof(double));
    MPI_Scatterv((rank==0?A_root:NULL), countsA, displsA, MPI_DOUBLE,
                 A_local, my_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // y_local
    double *y_local = (double*)malloc((size_t)my_rows * sizeof(double));
    for (int i = 0; i < my_rows; ++i) {
        double s = 0.0;
        const double *Ai = A_local + (size_t)i * N;
        for (int j = 0; j < N; ++j) s += Ai[j] * x[j];
        y_local[i] = s;
    }

    double t1 = MPI_Wtime();
    double local = t1 - t0;

    // y <- Gatherv по строкам
    MPI_Gatherv(y_local, my_rows, MPI_DOUBLE,
                (rank==0?y_root:NULL),
                (rank==0?countsY:NULL),
                (rank==0?displsY:NULL),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local, t_out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    free(x);
    free(A_local);
    free(y_local);
    if (rank == 0){
        free(countsA);
        free(displsA);
        free(countsY);
        free(displsY);
    }
}

// раздаём столбцы A и соответствующий кусок x, суммируем y_part
static void pack_cols_for_rank(const double *A, int N, int r, int size,
                               double **buf_out, int *cols_out)
{
    int base = N / size, rem = N % size;
    int col0 = r * base + (r < rem ? r : rem);
    int cols = base + (r < rem ? 1 : 0);

    double *buf = (double*)malloc((size_t)N * cols * sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < cols; ++c) {
            buf[(size_t)i * cols + c] = A[(size_t)i * N + (col0 + c)];
        }
    }
    *buf_out = buf;
    *cols_out = cols;
}

static void matvec_col_mpi(const double *A_root, const double *x_root, double *y_root, int N,
                           int rank, int size, double *t_out)
{
    int base = N / size, rem = N % size;
    int my_col0 = rank * base + (rank < rem ? rank : rem);
    int my_cols = base + (rank < rem ? 1 : 0);

    double *A_local = (double*)malloc((size_t)N * my_cols * sizeof(double));
    double *x_local = (double*)malloc((size_t)my_cols * sizeof(double));

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            double *bufA = NULL; int cols_r = 0;
            pack_cols_for_rank(A_root, N, r, size, &bufA, &cols_r);

            int col0_r = r * base + (r < rem ? r : rem);
            if (r == 0) {
                memcpy(A_local, bufA, (size_t)N * cols_r * sizeof(double));
                memcpy(x_local, x_root + col0_r, (size_t)cols_r * sizeof(double));
            } else {
                MPI_Send(bufA, N * cols_r, MPI_DOUBLE, r, 100, MPI_COMM_WORLD);
                MPI_Send(x_root + col0_r, cols_r, MPI_DOUBLE, r, 101, MPI_COMM_WORLD);
            }
            free(bufA);
        }
    } else {
        MPI_Recv(A_local, N * my_cols, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(x_local, my_cols, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // частичный вклад y_part (длина N)
    double *y_part = (double*)calloc((size_t)N, sizeof(double));
    for (int i = 0; i < N; ++i) {
        const double *Ai = A_local + (size_t)i * my_cols;
        double s = 0.0;
        for (int c = 0; c < my_cols; ++c) s += Ai[c] * x_local[c];
        y_part[i] = s;
    }

    double t1 = MPI_Wtime();
    double local = t1 - t0;

    // Суммируем вклады в y_root на корне
    MPI_Reduce(y_part, (rank==0 ? y_root : NULL), N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local, t_out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    free(A_local);
    free(x_local);
    free(y_part);
}

static void pack_block(const double *A, int N, int i0, int j0, int br, int bc, double *buf) {
    for (int r = 0; r < br; ++r) {
        const double *src = A + (size_t)(i0 + r) * N + j0;
        memcpy(buf + (size_t)r * bc, src, (size_t)bc * sizeof(double));
    }
}
static void matvec_block_mpi(const double *A_root, const double *x_root, double *y_root, int N,
                             int rank, int size, double *t_out)
{
    // Создаём квадратную решётку q×q
    int q = 0;
    while (q*q < size) ++q;
    if (q*q != size) {
        if (rank == 0) fprintf(stderr, "Для block требуется, чтобы число процессов было квадратом (q^2).\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    int Np = (N % q == 0) ? N : ( (N/q + 1) * q );


    int br = Np / q;
    int bc = Np / q;

    // Коммуникатор-решётка и подкоммуникаторы по строкам
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Comm grid; MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);

    int coords[2]; MPI_Cart_coords(grid, rank, 2, coords);
    int my_i = coords[0], my_j = coords[1];

    MPI_Comm row_comm;
    int remain_dims_row[2] = {0, 1};
    MPI_Cart_sub(grid, remain_dims_row, &row_comm);

    double *Ablk = (double*)malloc((size_t)br * bc * sizeof(double));
    double *xblk = (double*)malloc((size_t)bc * sizeof(double));
    double *y_part = (double*)calloc((size_t)br, sizeof(double));

    if (rank == 0) {
        for (int pi = 0; pi < q; ++pi) {
            for (int pj = 0; pj < q; ++pj) {
                int rnk; int ctmp[2] = {pi, pj};
                MPI_Cart_rank(grid, ctmp, &rnk);

                int i0 = pi * br;
                int j0 = pj * bc;

                double *tmp = (double*)calloc((size_t)br * bc, sizeof(double));
                for (int rr=0; rr<br; ++rr) {
                    int gi = i0 + rr;
                    if (gi >= N) continue;
                    for (int cc=0; cc<bc; ++cc) {
                        int gj = j0 + cc;
                        if (gj >= N) continue;
                        tmp[(size_t)rr*bc + cc] = A_root[(size_t)gi*N + gj];
                    }
                }

                double *tx = (double*)calloc((size_t)bc, sizeof(double));
                for (int cc=0; cc<bc; ++cc) {
                    int gj = j0 + cc;
                    if (gj < N) tx[cc] = x_root[gj];
                }

                if (rnk == 0) {
                    memcpy(Ablk, tmp, (size_t)br*bc*sizeof(double));
                    memcpy(xblk, tx, (size_t)bc*sizeof(double));
                } else {
                    MPI_Send(tmp, br*bc, MPI_DOUBLE, rnk, 200, MPI_COMM_WORLD);
                    MPI_Send(tx, bc, MPI_DOUBLE, rnk, 201, MPI_COMM_WORLD);
                }
                free(tmp); free(tx);
            }
        }
    } else {
        MPI_Recv(Ablk, br*bc, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(xblk, bc, MPI_DOUBLE, 0, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int r = 0; r < br; ++r) {
        double s = 0.0;
        const double *rowA = Ablk + (size_t)r * bc;
        for (int c = 0; c < bc; ++c) s += rowA[c] * xblk[c];
        y_part[r] = s;
    }

    double *y_row_sum = NULL;
    if (my_j == 0) y_row_sum = (double*)calloc((size_t)br, sizeof(double));

    MPI_Reduce(y_part, y_row_sum, br, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    double t1 = MPI_Wtime();
    double local = t1 - t0;

    if (my_j == 0) {
        if (rank == 0) {
            for (int i=0;i<N;i++) y_root[i]=0.0;

            int i0 = my_i * br;
            for (int r=0; r<br; ++r) {
                int gi = i0 + r;
                if (gi < N) y_root[gi] = y_row_sum[r];
            }

            for (int pi = 0; pi < q; ++pi) {
                int ctmp[2] = {pi, 0};
                int rnk; MPI_Cart_rank(grid, ctmp, &rnk);
                if (rnk == 0) continue;
                double *buf = (double*)malloc((size_t)br * sizeof(double));
                MPI_Recv(buf, br, MPI_DOUBLE, rnk, 210, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int i0p = pi * br;
                for (int r=0; r<br; ++r) {
                    int gi = i0p + r;
                    if (gi < N) y_root[gi] += buf[r];
                }
                free(buf);
            }
        } else {
            MPI_Send(y_row_sum, br, MPI_DOUBLE, 0, 210, MPI_COMM_WORLD);
        }
    }

    MPI_Reduce(&local, t_out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (y_row_sum) free(y_row_sum);
    free(Ablk); free(xblk); free(y_part);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&grid);
}
