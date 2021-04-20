
#define _GNU_SOURCE
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#include <numa.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <cblas.h>

static int rma_bench_create_comm(MPI_Comm *inter_node_numa_comm, MPI_Comm *intra_numa_comm)
{
    int rank, mpi_errno = MPI_SUCCESS;
    MPI_Comm node_comm;
    int cpu = sched_getcpu();
    int local_numa_id = numa_node_of_cpu(cpu);
    int num_numa_node = numa_num_task_nodes();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpi_errno = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    if (mpi_errno != MPI_SUCCESS)
    {
        printf("[%d] MPI_Comm_split_type error, mpi_errno %d\n", __LINE__, mpi_errno);
        fflush(stdout);
        exit(1);
    }

    MPI_Comm_split(MPI_COMM_WORLD, local_numa_id, rank, inter_node_numa_comm);
    if (mpi_errno != MPI_SUCCESS)
    {
        printf("[%d] MPI_Comm_split error, mpi_errno %d\n", __LINE__, mpi_errno);
        fflush(stdout);
        exit(1);
    }

    MPI_Comm_split(node_comm, local_numa_id, rank, intra_numa_comm);
    if (mpi_errno != MPI_SUCCESS)
    {
        printf("[%d] MPI_Comm_split_type error, mpi_errno %d\n", __LINE__, mpi_errno);
        fflush(stdout);
        exit(1);
    }

    MPI_Comm_free(&node_comm);

    return mpi_errno;
}

#define BUF_INIT_VALUE 0x01

static int check_results(char *intra_buf, char *inter_buf, int buf_sz)
{
    int err = 0;
    for (int i = 0; i < buf_sz; ++i)
    {
        if ((intra_buf[i] & 0xff) != BUF_INIT_VALUE)
        {
            printf("intra_buf error %x at %d, it should be %x\n", intra_buf[i] & 0xff, i, BUF_INIT_VALUE);
            err++;
        }

        if ((inter_buf[i] & 0xff) != BUF_INIT_VALUE)
        {
            printf("inter_buf error %x at %d, it should be %x\n", intra_buf[i] & 0xff, i, BUF_INIT_VALUE);
            err++;
        }
    }
    memset(intra_buf, 0, buf_sz);
    memset(inter_buf, 0, buf_sz);
    return err;
}

void init_matrix(double **a, double **b, double **c, int dim, int wcnt) {
    int total_elem = dim * dim * wcnt;
    assert(total_elem >= 0);
    *a = (double *) malloc(sizeof(double) * total_elem);
    *b = (double *) malloc(sizeof(double) * total_elem);
    *c = (double *) malloc(sizeof(double) * total_elem);

    double *atmp = *a;
    double *btmp = *b;
    double *ctmp = *c;

    srand(0);

    for(int i = 0 ; i < total_elem; ++i) {
        atmp[i] = (double) (rand() % 256) + (double) (rand() % 256) / 1e3;
        btmp[i] = (double) (rand() % 256) + (double) (rand() % 256) / 1e3;
        ctmp[i] = (double) (rand() % 256) + (double) (rand() % 256) / 1e3;
    }
}

/* In this benchmark, we assume there are two nodes; each node has the same number
 * of processes and processes with same local rank bind to the same local core id. */
#define WARMUP_ITERATION 6
#define AVE_ITERATION 9

#define WORKER_PER_NUMA 18
#define MATRIX_PER_WORKER 9
#define COMMPROC_PER_NUMA 18

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    if (argc < 4)
    {
        printf("[Usage] mpirun -n ${np} ./bin block iteration cblas_time\n");
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm inter_node_numa_comm, intra_numa_comm;
    rma_bench_create_comm(&inter_node_numa_comm, &intra_numa_comm);

    int block = atoi(argv[1]); // message size
    int message_cnt_s1 = atoi(argv[2]); // message cnt per communication proc in stage 1
    int message_cnt_s2 = atoi(argv[3]); // message cnt per communication proc in stage 2
    int dim = atoi(argv[4]); // matrix dimension

    // char hostname[64];
    // gethostname(hostname, 64);
    // printf("rank %d - hostname %s\n", rank ,hostname);

    int intra_numa_size;
    int intra_numa_rank, inter_numa_rank;

    MPI_Comm_size(intra_numa_comm, &intra_numa_size);

    MPI_Comm_rank(intra_numa_comm, &intra_numa_rank);
    MPI_Comm_rank(inter_node_numa_comm, &inter_numa_rank);
    int step = intra_numa_size >> 1;

    int actual_message_cnt_s1, buf_sz_s1;
    int actual_message_cnt_s2, buf_sz_s2;
    int actual_wcnt, boundary, inter_peer;
    double *a, *b, *c;

    /* get #messages to be issued */
    int total_message_cnt_s1 = message_cnt_s1 * COMMPROC_PER_NUMA;
    int total_message_cnt_s2 = message_cnt_s2 * COMMPROC_PER_NUMA;

    actual_message_cnt_s1 = total_message_cnt_s1 * (intra_numa_rank + 1) / intra_numa_size - total_message_cnt_s1 * intra_numa_rank / intra_numa_size;
    buf_sz_s1 = block * actual_message_cnt_s1;

    actual_message_cnt_s2 = total_message_cnt_s2 * (intra_numa_rank + 1) / intra_numa_size - total_message_cnt_s2 * intra_numa_rank / intra_numa_size;
    buf_sz_s2 = block * actual_message_cnt_s2;

    /* get workload to be done  */
    int total_wcnt = MATRIX_PER_WORKER * WORKER_PER_NUMA;
    actual_wcnt = total_wcnt * (intra_numa_rank + 1) / intra_numa_size - total_wcnt * intra_numa_rank / intra_numa_size ;

    init_matrix(&a, &b, &c, dim, actual_wcnt);

    void *inter_win_ptr_s1, *inter_win_ptr_s2;
    MPI_Win inter_numa_win_s1, inter_numa_win_s2;
    MPI_Win_allocate(buf_sz_s1, 1, MPI_INFO_NULL, inter_node_numa_comm, (void *)&inter_win_ptr_s1, &inter_numa_win_s1);

    memset(inter_win_ptr_s1, BUF_INIT_VALUE, buf_sz_s1);
    MPI_Barrier(MPI_COMM_WORLD);

    char *inter_local_buf_s1 = malloc(buf_sz_s1);
    memset(inter_local_buf_s1, 0, buf_sz_s1);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, inter_numa_win_s1);

    /***********/
    /* stage 1 */
    /***********/
    /* warm up */
    for (int k = 0; k < WARMUP_ITERATION; ++k)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (intra_numa_rank < boundary) {
            /* do shared and inter-node rma */
            int inter_peer;

            if(inter_numa_rank < intra_numa_size) {
                inter_peer = inter_numa_rank + intra_numa_size;

                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);

                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);

            } else {
                inter_peer = inter_numa_rank - intra_numa_size;
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);

                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);
            }

            //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
        }
        else {
            if(inter_numa_rank >= intra_numa_size) {
                inter_peer = inter_numa_rank - intra_numa_size;

                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);

                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);

            } else {
                inter_peer = inter_numa_rank + intra_numa_size;
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);

                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);
            }
        }

        // memset(intra_local_buf, 0, buf_sz);
        // memset(inter_local_buf, 0, buf_sz);
    }

    double time = 0.0;
    double comm_time = 0.0, cblas_time = 0.0;
    double max_time;
    double max_comm, max_cblas;
    double rec_times[AVE_ITERATION];
    double rec_comm_times[AVE_ITERATION];
    double rec_cblas_times[AVE_ITERATION];
    /***********/
    /* stage 1 */
    /***********/
    /* execution */
    for (int k = 0; k < AVE_ITERATION; ++k)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime();
        if (intra_numa_rank < boundary) {
            /* do shared and inter-node rma */
            int inter_peer;

            if(inter_numa_rank < intra_numa_size) {
                inter_peer = inter_numa_rank + intra_numa_size;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);
                comm_time = MPI_Wtime() - comm_time;

                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

            } else {
                inter_peer = inter_numa_rank - intra_numa_size;
                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);
                comm_time = MPI_Wtime() - comm_time;
            }

            //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
        }
        else {
            if(inter_numa_rank >= intra_numa_size) {
                inter_peer = inter_numa_rank - intra_numa_size;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);
                comm_time = MPI_Wtime() - comm_time;

                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

            } else {
                inter_peer = inter_numa_rank + intra_numa_size;

                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s1; i += block)
                    MPI_Get(inter_local_buf_s1 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, inter_numa_win_s1);
                MPI_Win_flush_all(inter_numa_win_s1);
                comm_time = MPI_Wtime() - comm_time;
            }
        }

        time = MPI_Wtime() - time;
        MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&cblas_time, &max_cblas, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        rec_comm_times[k] = max_comm;
        rec_cblas_times[k] = max_cblas;
        rec_times[k] = max_time;
    }

    if (rank == 0)
    {
        double ave_time = 0.0, ave_comm_time = 0.0, ave_cblas_time = 0.0;
        for (int k = 0; k < AVE_ITERATION; ++k)
        {
            ave_time += rec_times[k];
            ave_comm_time += rec_comm_times[k];
            ave_cblas_time += rec_cblas_times[k];
            //printf("Iter %d - total time %.3lf, comm %.3lf, sleep %.3lf\n", k, rec_times[k] * 1e6, rec_comm_times[k] * 1e6, rec_cblas_times[k] * 1e6);
        }

        ave_time = ave_time / AVE_ITERATION;
        ave_comm_time = ave_comm_time / AVE_ITERATION;
        ave_cblas_time = ave_cblas_time / AVE_ITERATION;
        printf("1 %.3lf %.3lf %.3lf\n", ave_time * 1e6, ave_comm_time * 1e6, ave_cblas_time * 1e6);
    }

    MPI_Win_unlock_all(inter_numa_win_s1);
    MPI_Barrier(MPI_COMM_WORLD);

    free(inter_local_buf_s1);
    MPI_Win_free(&inter_numa_win_s1);

    /**********************************/
    /* stage 2 */
    /**********************************/
    /* warm up run */
    char *inter_local_buf_s2 = malloc(buf_sz_s2);
    memset(inter_local_buf_s2, 0, buf_sz_s2);

    MPI_Win_allocate(buf_sz_s2, 1, MPI_INFO_NULL, inter_node_numa_comm, (void *)&inter_win_ptr_s2, &inter_numa_win_s2);
    memset(inter_win_ptr_s2, BUF_INIT_VALUE, buf_sz_s2);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, inter_numa_win_s2);
    /* warm up run */
    for (int k = 0; k < AVE_ITERATION; ++k)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime();
        if (intra_numa_rank < boundary) {
            /* do shared and inter-node rma */
            int inter_peer;

            if(inter_numa_rank < intra_numa_size) {
                inter_peer = inter_numa_rank + intra_numa_size;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s2; i += block)
                    MPI_Accumulate(inter_local_buf_s2 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, MPI_SUM, inter_numa_win_s2);
                MPI_Win_flush_all(inter_numa_win_s2);
                comm_time = MPI_Wtime() - comm_time;

                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

            } else {
                inter_peer = inter_numa_rank - intra_numa_size;
                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s2; i += block)
                    MPI_Accumulate(inter_local_buf_s2 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, MPI_SUM, inter_numa_win_s2);
                MPI_Win_flush_all(inter_numa_win_s2);
                comm_time = MPI_Wtime() - comm_time;
            }

            //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
        }
        else {
            if(inter_numa_rank >= intra_numa_size) {
                inter_peer = inter_numa_rank - intra_numa_size;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s2; i += block)
                    MPI_Accumulate(inter_local_buf_s2 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, MPI_SUM, inter_numa_win_s2);
                MPI_Win_flush_all(inter_numa_win_s2);
                comm_time = MPI_Wtime() - comm_time;

                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

            } else {
                inter_peer = inter_numa_rank + intra_numa_size;

                cblas_time = MPI_Wtime();
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

                comm_time = MPI_Wtime();
                for (int i = 0; i < buf_sz_s2; i += block)
                    MPI_Accumulate(inter_local_buf_s2 + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, MPI_SUM, inter_numa_win_s2);
                MPI_Win_flush_all(inter_numa_win_s2);
                comm_time = MPI_Wtime() - comm_time;
            }
        }

        time = MPI_Wtime() - time;
        MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&cblas_time, &max_cblas, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        rec_comm_times[k] = max_comm;
        rec_cblas_times[k] = max_cblas;
        rec_times[k] = max_time;
    }

    free(inter_local_buf_s2);
    MPI_Win_free(&inter_numa_win_s2);
    MPI_Comm_free(&intra_numa_comm);
    MPI_Comm_free(&inter_node_numa_comm);
    MPI_Finalize();
    return 0;
}
