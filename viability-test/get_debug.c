#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void progress_stealing_test(int rank, int buf_sz){
    /* assume 0, 1 are on the same node, 2 is on different node */
    void *winbuf;
    MPI_Win win;

    MPI_Win_allocate(buf_sz, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winbuf, &win);
    memset(winbuf, 0, buf_sz);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    if(rank == 0){
        sleep(0.5);
    }else if(rank == 1){
        int flag = 0;
        double time = MPI_Wtime();
        MPI_Get(winbuf, buf_sz, MPI_CHAR, 0, 0, buf_sz, MPI_CHAR, win);
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        if(flag){
            printf("find a message\n");
        }
        MPI_Win_flush(0, win);
        time = MPI_Wtime() - time;
        
        printf("%d %.3lf\n", buf_sz, time * 1e6);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    return;
}

#define ITERATION 1000

int main(int argc, char *argv[]){
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buf_sz = atoi(argv[1]);
    // char hostname[64];
    // gethostname(hostname, 64);
    // printf("rank %d - hostname %s\n", rank ,hostname);
    for(int i = 0; i< ITERATION; ++i)
        progress_stealing_test(rank, buf_sz);
    
    
    MPI_Finalize();
    return 0;
}