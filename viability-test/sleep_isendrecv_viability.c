#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
/* assume 0, 2 are on the same node, 1, 3 is on different node
 * Rank 0 issues isend to rank 1, and rank 1 performs irecv and sleep.
 */
void progress_stealing_test(int rank, int buf_sz, int sleep_time, int iter){
    char *buf;
    MPI_Win win;

    buf = (char *) malloc(buf_sz);
    memset(buf, rank, buf_sz);
    MPI_Barrier(MPI_COMM_WORLD);
    
    int origin = 0;
    int target = 1;
    /* warm up */
    if(rank == target) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Irecv(buf, buf_sz, MPI_CHAR, origin, 0, MPI_COMM_WORLD, &reqs[i]);
        usleep(sleep_time);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
    }else if(rank == origin) {
        for(int i = 0; i < iter; ++i)
            MPI_Send(buf, buf_sz, MPI_CHAR, target, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double tcomm;
    double total = MPI_Wtime();
    if(rank == target) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Irecv(buf, buf_sz, MPI_CHAR, origin, 0, MPI_COMM_WORLD, &reqs[i]);
        usleep(sleep_time);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
    }else if(rank == origin) {
        tcomm = MPI_Wtime();
        for(int i = 0; i < iter; ++i)
            MPI_Send(buf, buf_sz, MPI_CHAR, target, 0, MPI_COMM_WORLD);
        tcomm = MPI_Wtime() - tcomm;
    }
    total = MPI_Wtime() - total;
    
    /* other processes do nothing */
    MPI_Barrier(MPI_COMM_WORLD);
    int max_total;
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("%d %.3lf %.3lf %d\n", buf_sz, max_total * 1e6, tcomm * 1e6, sleep_time);
    }

    if(rank == target){
        for(int i = 0; i < buf_sz; ++i) {
            if(buf[i] != origin) {
                printf("Error: winbuf[%d] = %d is not %d\n", i, buf[i], origin);
            }
        }
    }
    return;
}

int main(int argc, char *argv[]){
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buf_sz = atoi(argv[1]);
    int sleep_time = atoi(argv[2]);
    int iter = atoi(argv[3]);
    // char hostname[64];
    // gethostname(hostname, 64);
    // printf("rank %d - hostname %s\n", rank ,hostname);
    // fflush(stdout);
    // sleep(10);
    
    progress_stealing_test(rank, buf_sz, sleep_time, iter);
    
    MPI_Finalize();
    return 0;
}