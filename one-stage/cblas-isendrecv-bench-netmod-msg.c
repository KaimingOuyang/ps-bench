
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

#define WORKER_PER_NUMA 9
#define WMATRIX_PER_WORKER 9
#define COMMPROC_PER_NUMA 9

#define MSG_CNT 15
int msg_size[15] = {16,32,64,128,512,1024,8192,16384,65536,131072,524288,1048576,4194304,8388608,16777216};

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    if (argc < 3)
    {
        printf("[Usage] mpirun -n ${np} ./bin message_cnt cblas_time\n");
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm inter_node_numa_comm, intra_numa_comm;
    rma_bench_create_comm(&inter_node_numa_comm, &intra_numa_comm);

    //int block = atoi(argv[1]); // message size
    int message_cnt = atoi(argv[1]); // message cnt per communication proc
    int dim = atoi(argv[2]); // matrix dimension

    // char hostname[64];
    // gethostname(hostname, 64);
    // printf("rank %d - hostname %s\n", rank ,hostname);

    int intra_numa_size;
    int intra_numa_rank, inter_numa_rank;

    MPI_Comm_size(intra_numa_comm, &intra_numa_size);

    MPI_Comm_rank(intra_numa_comm, &intra_numa_rank);
    MPI_Comm_rank(inter_node_numa_comm, &inter_numa_rank);
    int step = COMMPROC_PER_NUMA;

    for(int iter = 0; iter<MSG_CNT; ++iter) {
        int actual_message_cnt, buf_sz, actual_wcnt;
        double *a, *b, *c;

        int block = msg_size[iter];
        /* get #messages to be issued */
        int total_message_cnt = message_cnt * COMMPROC_PER_NUMA;

        if(intra_numa_rank < step) {
            int cur_commproc = COMMPROC_PER_NUMA;
            actual_message_cnt = total_message_cnt * (intra_numa_rank + 1) / cur_commproc - total_message_cnt * intra_numa_rank / cur_commproc;
            buf_sz = block * actual_message_cnt;
            // printf("rank %d, numa rank %d - total message %d, issue %d\n", rank, intra_numa_rank, total_message_cnt, actual_message_cnt);
        } else {
            /* get workload to be done  */
            int total_wcnt = WMATRIX_PER_WORKER * WORKER_PER_NUMA;
            int cur_worker = intra_numa_size - step;
            int wrank = intra_numa_rank - step;
            actual_wcnt = total_wcnt * (wrank + 1) / cur_worker - total_wcnt * wrank / cur_worker;

            init_matrix(&a, &b, &c, dim, actual_wcnt);
            buf_sz = block * (total_message_cnt / step + 1);
            // printf("rank %d, numa rank %d - total wcnt %d, my wcnt %d\n", rank, intra_numa_rank, total_wcnt, actual_wcnt);
        }

        char *inter_local_buf = malloc(buf_sz);
        memset(inter_local_buf, 0, buf_sz);

        int num_numa_worker = intra_numa_size - step;

        for (int k = 0; k < WARMUP_ITERATION; ++k)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (intra_numa_rank < step) {
                /* do shared and inter-node rma */
                int inter_peer = step + intra_numa_rank % num_numa_worker;
                if (inter_numa_rank < intra_numa_size) {
                    inter_peer += intra_numa_size;
                }

                MPI_Request *sreqs = (MPI_Request*) malloc(sizeof(MPI_Request) * actual_message_cnt);
                for (int i = 0; i < actual_message_cnt; i++) {
                    MPI_Isend(inter_local_buf + i * block, block, MPI_CHAR, inter_peer, 0, inter_node_numa_comm, &sreqs[i]);
                }
                MPI_Waitall(actual_message_cnt, sreqs, MPI_STATUSES_IGNORE);
                free(sreqs);
                //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
            }
            else {
                int total_sender = step / num_numa_worker + 1;
                int sender_cnt = 0;
                MPI_Request **rreqs;
                rreqs = malloc(sizeof(MPI_Request *) * total_sender);
                for(int sender = 0; sender < step; ++sender){
                    int intra_dest = step + sender % num_numa_worker;
                    if(intra_dest == intra_numa_rank) {
                        int inter_peer = sender;
                        if (inter_numa_rank < intra_numa_size){
                            inter_peer = intra_numa_size;
                        }
                        rreqs[sender_cnt++] = (MPI_Request*) malloc(sizeof(MPI_Request) * actual_message_cnt);
                        for (int i = 0; i < actual_message_cnt; i++)
                            MPI_Irecv(inter_local_buf + i * block, block, MPI_CHAR, inter_peer, 0, inter_node_numa_comm, &rreqs[sender_cnt-1][i]);
                    }   
                }
                    
                /* do dummy compute */
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);

                for(int sender_id = 0; sender_id < sender_cnt; ++sender_id){
                        MPI_Waitall(actual_message_cnt, rreqs[sender_id], MPI_STATUSES_IGNORE);
                        free(rreqs[sender_id]);
                }
                free(rreqs);
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
            if (intra_numa_rank < step) {
                /* do shared and inter-node rma */
                int inter_peer = step + intra_numa_rank % num_numa_worker;
                if (inter_numa_rank < intra_numa_size) {
                    inter_peer += intra_numa_size;
                }

                MPI_Request *sreqs = (MPI_Request*) malloc(sizeof(MPI_Request) * actual_message_cnt);
                for (int i = 0; i < actual_message_cnt; i++) {
                    MPI_Isend(inter_local_buf + i * block, block, MPI_CHAR, inter_peer, 0, inter_node_numa_comm, &sreqs[i]);
                }
                MPI_Waitall(actual_message_cnt, sreqs, MPI_STATUSES_IGNORE);
                free(sreqs);
                //assert(check_results(intra_local_buf, inter_local_buf, buf_sz) == 0);
            }
            else {
                int total_sender = step / num_numa_worker + 1;
                int sender_cnt = 0;
                MPI_Request **rreqs;
                rreqs = malloc(sizeof(MPI_Request *) * total_sender);
                for(int sender = 0; sender < step; ++sender){
                    int intra_dest = step + sender % num_numa_worker;
                    if(intra_dest == intra_numa_rank) {
                        int inter_peer = sender;
                        if (inter_numa_rank < intra_numa_size){
                            inter_peer = intra_numa_size;
                        }
                        rreqs[sender_cnt++] = (MPI_Request*) malloc(sizeof(MPI_Request) * actual_message_cnt);
                        for (int i = 0; i < actual_message_cnt; i++)
                            MPI_Irecv(inter_local_buf + i * block, block, MPI_CHAR, inter_peer, 0, inter_node_numa_comm, &rreqs[sender_cnt-1][i]);
                    }   
                }
                
                cblas_time = MPI_Wtime();
                /* do dummy compute */
                for(int i = 0; i < actual_wcnt; ++i)
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, a + i * dim * dim, dim, b + i * dim * dim, dim, 1.0, c + i * dim * dim, dim);
                cblas_time = MPI_Wtime() - cblas_time;

                comm_time = MPI_Wtime();
                for(int sender_id = 0; sender_id < sender_cnt; ++sender_id){
                        MPI_Waitall(actual_message_cnt, rreqs[sender_id], MPI_STATUSES_IGNORE);
                        free(rreqs[sender_id]);
                }
                comm_time = MPI_Wtime() - comm_time;
                free(rreqs);
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
            /*
            for (int k = 0; k < AVE_ITERATION; ++k)
            {
                time_devi += (rec_times[k] - ave_time) * (rec_times[k] - ave_time);
            }
            time_devi = sqrt(time_devi / AVE_ITERATION);
            */
            printf("%d %.3lf %.3lf %.3lf\n", block, ave_time * 1e6, ave_comm_time * 1e6, ave_cblas_time * 1e6);
        }

        free(inter_local_buf);
        if(intra_numa_rank >= step) {
            free(a);
            free(b);
            free(c);
        }
    }

    MPI_Comm_free(&intra_numa_comm);
    MPI_Comm_free(&inter_node_numa_comm);
    MPI_Finalize();
    return 0;
}
