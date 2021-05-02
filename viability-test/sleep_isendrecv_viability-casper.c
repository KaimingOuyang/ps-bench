#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
/* assume 0, 1, 2 and 3 are on the same node
 * Rank 0 issues isend to rank 1, and rank 1 performs irecv and sleep.
 */
int stride = 64;
void progress_stealing_test(int rank, int buf_sz, int sleep_time, int iter) {
    char *buf;
    MPI_Win win;
    MPI_Comm shm_comm, comm_world;
    MPI_Info info = MPI_INFO_NULL;
    int total_buf_sz;

    int origin = 0;
    int target = 1;

    if(rank == origin || rank == target) {
        total_buf_sz = buf_sz << stride;
    }
    else {
        total_buf_sz = 0;
    }

    MPI_Datatype recv_type;
    MPI_Type_vector(buf_sz, 1, stride, MPI_CHAR, &recv_type);
    MPI_Type_commit(&recv_type);

    MPI_Info_create(&info);
    MPI_Info_set(info, (char *) "shmbuf_regist", (char *) "true");

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, info, &shm_comm);
    MPI_Info_free(&info);

    MPI_Win_allocate_shared(total_buf_sz, 1, MPI_INFO_NULL, shm_comm, &buf, &win);

    memset(buf, rank, total_buf_sz);

    MPI_Info_create(&info);
    MPI_Info_set(info, (char *) "wildcard_used", (char *) "none");
    MPI_Info_set(info, (char *) "datatype_used", (char *) "predefined");
    MPI_Comm_dup_with_info(MPI_COMM_WORLD, info, &comm_world);
    MPI_Info_free(&info);

    MPI_Barrier(comm_world);

    /* warm up */
    if(rank == target) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Irecv(buf, 1, recv_type, origin, 0, comm_world, &reqs[i]);
        usleep(sleep_time);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
    } else if(rank == origin) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Isend(buf, buf_sz, MPI_CHAR, target, 0, comm_world, &reqs[i]);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(comm_world);

    double tcomm;
    double total = MPI_Wtime();
    if(rank == target) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Irecv(buf, 1, recv_type, origin, 0, comm_world, &reqs[i]);
        usleep(sleep_time);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
    } else if(rank == origin) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        tcomm = MPI_Wtime();
        for(int i = 0; i < iter; ++i)
            MPI_Isend(buf, buf_sz, MPI_CHAR, target, 0, comm_world, &reqs[i]);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
        tcomm = MPI_Wtime() - tcomm;
    }
    total = MPI_Wtime() - total;

    /* other processes do nothing */
    MPI_Barrier(comm_world);
    double max_total;
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm_world);
    if(rank == 0) {
        printf("%d %.3lf %.3lf %d\n", buf_sz, max_total * 1e6, tcomm * 1e6, sleep_time);
    }

    if(rank == target) {
        for(int i = 0; i < buf_sz; i += stride) {
            if(buf[i] != origin) {
                printf("Error: winbuf[%d] = %d is not %d\n", i, buf[i], origin);
            }
        }
    }
    MPI_Win_free(&win);
    MPI_Comm_free(&comm_world);
    MPI_Comm_free(&shm_comm);
    MPI_Type_free(&recv_type);
    return;
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buf_sz = atoi(argv[1]);
    int sleep_time = atoi(argv[2]);
    int iter = atoi(argv[3]);
    if(argc == 5) {
        stride = atoi(argv[4]);
    }
    // char hostname[64];
    // gethostname(hostname, 64);
    // printf("rank %d - hostname %s\n", rank ,hostname);
    // fflush(stdout);
    // sleep(10);

    progress_stealing_test(rank, buf_sz, sleep_time, iter);

    MPI_Finalize();
    return 0;
}