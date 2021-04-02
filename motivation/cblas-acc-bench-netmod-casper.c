
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

#define BUF_INIT_VALUE 0xff

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

void init_matrix(double **a, double **b, double **c, int dim) {
    *a = (double *) malloc(sizeof(double) * dim * dim);
    *b = (double *) malloc(sizeof(double) * dim * dim);
    *c = (double *) malloc(sizeof(double) * dim * dim);

    double *atmp = *a;
    double *btmp = *b;
    double *ctmp = *c;

    srand(0);

    for(int i = 0 ; i<dim*dim; ++i) {
        atmp[i] = (double) (rand() % 256) + (double) (rand() % 256) / 1e3;
        btmp[i] = (double) (rand() % 256) + (double) (rand() % 256) / 1e3;
        ctmp[i] = (double) (rand() % 256) + (double) (rand() % 256) / 1e3;
    }
}

/* In this benchmark, we assume there are two nodes; each node has the same number
 * of processes and processes with same local rank bind to the same local core id. */
#define WARMUP_ITERATION 10
#define AVE_ITERATION 10

#define WORKER_PER_NUMA 2

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

    int block = atoi(argv[1]);
    int iteration = atoi(argv[2]);
    int dim = atoi(argv[3]); // matrix dimension
    int buf_sz = block * iteration;

    /* buf_sz should be multiple of block */
    double *a, *b, *c;
    init_matrix(&a, &b, &c, dim);

    /* we only allow even number of workers */
    // char hostname[64];
    // gethostname(hostname, 64);
    // printf("rank %d - hostname %s\n", rank ,hostname);

    int intra_numa_size;
    int intra_numa_rank, inter_numa_rank;

    MPI_Comm_size(intra_numa_comm, &intra_numa_size);

    MPI_Comm_rank(intra_numa_comm, &intra_numa_rank);
    MPI_Comm_rank(inter_node_numa_comm, &inter_numa_rank);
    int step = intra_numa_size >> 1;
    double cur_worker = intra_numa_size / 2 + (intra_numa_size % 2  == 0 ? 0 : 1);
    int total_ct = WORKER_PER_NUMA / cur_worker * (intra_numa_rank + 1) - WORKER_PER_NUMA / cur_worker * intra_numa_rank;

    void *inter_win_ptr;
    MPI_Win inter_numa_win;
    MPI_Win_allocate(buf_sz, 1, MPI_INFO_NULL, inter_node_numa_comm, (void *)&inter_win_ptr, &inter_numa_win);

    memset(inter_win_ptr, BUF_INIT_VALUE, buf_sz);
    MPI_Barrier(MPI_COMM_WORLD);

    char *inter_local_buf = malloc(buf_sz);
    memset(inter_local_buf, 0, buf_sz);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, inter_numa_win);

    for (int k = 0; k < WARMUP_ITERATION; ++k)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (intra_numa_rank < step)
        {
            /* do shared and inter-node rma */
            int inter_peer;
            if (inter_numa_rank < intra_numa_size)
                inter_peer = inter_numa_rank + intra_numa_size;
            else
                inter_peer = inter_numa_rank - intra_numa_size;
            inter_peer += step;

            for (int i = 0; i < buf_sz; i += block)
            {
                //MPI_Get(intra_local_buf + i, block, MPI_CHAR, share_peer, i, block, MPI_CHAR, intra_numa_win);
                MPI_Accumulate(inter_local_buf + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, MPI_BXOR, inter_numa_win);
            }
            //MPI_Win_flush_all(intra_numa_win);
            MPI_Win_flush_all(inter_numa_win);
            //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
        }
        else
        {
            /* do dummy compute */
            for(int i = 0; i<total_ct;++i)
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a, dim, b, dim, 1.0, c, dim);
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

    for (int k = 0; k < AVE_ITERATION; ++k)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime();
        if (intra_numa_rank < step)
        {
            /* do shared and inter-node rma */
            comm_time = MPI_Wtime();
            int inter_peer;
            if (inter_numa_rank < intra_numa_size)
                inter_peer = inter_numa_rank + intra_numa_size;
            else
                inter_peer = inter_numa_rank - intra_numa_size;
            inter_peer += step;

            for (int i = 0; i < buf_sz; i += block)
            {
                MPI_Accumulate(inter_local_buf + i, block, MPI_CHAR, inter_peer, i, block, MPI_CHAR, MPI_BXOR, inter_numa_win);
            }
            MPI_Win_flush_all(inter_numa_win);
            //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
            comm_time = MPI_Wtime() - comm_time;
        }
        else
        {
            /* do dummy compute */
            cblas_time = MPI_Wtime();
            for(int i = 0; i<total_ct;++i)
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a, dim, b, dim, 1.0, c, dim);
            cblas_time = MPI_Wtime() - cblas_time;
        }

        time = MPI_Wtime() - time;
        MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&cblas_time, &max_cblas, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        rec_comm_times[k] = max_comm;
        rec_cblas_times[k] = max_cblas;
        rec_times[k] = max_time;
    }

    MPI_Win_unlock_all(inter_numa_win);

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
        /*
        for (int k = 0; k < AVE_ITERATION; ++k)
        {
            time_devi += (rec_times[k] - ave_time) * (rec_times[k] - ave_time);
        }
        time_devi = sqrt(time_devi / AVE_ITERATION);
        */
        printf("%d %.3lf %.3lf %.3lf\n", block, ave_time * 1e6, ave_comm_time * 1e6, ave_cblas_time * 1e6);
    }

    MPI_Win_free(&inter_numa_win);
    MPI_Comm_free(&intra_numa_comm);
    MPI_Comm_free(&inter_node_numa_comm);
    MPI_Finalize();
    return 0;
}
