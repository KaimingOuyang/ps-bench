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
    if(rank == 1) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Irecv(buf, buf_sz, MPI_CHAR, origin, 0, MPI_COMM_WORLD, &reqs[i]);
        usleep(sleep_time);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
    }else if(rank == 0) {
        for(int i = 0; i < iter; ++i)
            MPI_Send(buf, buf_sz, MPI_CHAR, target, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 1) {
        MPI_Request *reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * iter);
        for(int i = 0; i < iter; ++i)
            MPI_Irecv(buf, buf_sz, MPI_CHAR, origin, 0, MPI_COMM_WORLD, &reqs[i]);
        usleep(sleep_time);
        MPI_Waitall(iter, reqs, MPI_STATUSES_IGNORE);
        for(int i = 0; i < buf_sz; ++i){
            if(buf[i] != origin){
                printf("Error: buf[%d] = %d is not %d\n", i, buf[i], origin);
            }
        }
    }else if(rank == 0) {
        double time = MPI_Wtime();
        for(int i = 0; i < iter; ++i)
            MPI_Send(buf, buf_sz, MPI_CHAR, target, 0, MPI_COMM_WORLD);
        time = MPI_Wtime() - time;
        
        printf("%d %.3lf\n", buf_sz, time * 1e6);
    }
    
    /* other processes do nothing */
    MPI_Barrier(MPI_COMM_WORLD);

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