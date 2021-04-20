#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
/* assume 0, 2 are on the same node, 1, 3 is on different node
 * Rank 0 issues MPI_Put to rank 1, and rank 1 performs sleep.
 */

void progress_stealing_test(int rank, int buf_sz, int sleep_time, int iter) {
    char *winbuf;
    MPI_Win win;

    MPI_Win_allocate(buf_sz, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &winbuf, &win);
    memset(winbuf, rank, buf_sz);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    int origin = 0;
    int target = 1;
    /* warm up */
    if(rank == target) {        
        usleep(sleep_time);
    } else if(rank == origin) {
        for(int i = 0; i < iter; ++i)
            MPI_Put(winbuf, buf_sz, MPI_CHAR, target, 0, buf_sz, MPI_CHAR, win);
        MPI_Win_flush_all(win);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double tcomm;
    double total = MPI_Wtime();
    if(rank == target) {
        usleep(sleep_time);
    } else if(rank == origin) {
        tcomm = MPI_Wtime();
        for(int i = 0; i < iter; ++i)
            MPI_Put(winbuf, buf_sz, MPI_CHAR, target, 0, buf_sz, MPI_CHAR, win);
        MPI_Win_flush_all(win);
        tcomm = MPI_Wtime() - tcomm;
    }
    total = MPI_Wtime() - total;

    /* other processes do nothing */
    MPI_Barrier(MPI_COMM_WORLD);
    int max_total;
    MPI_Reduce(&total, &max_total, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("%d %.3lf %.3lf %d\n", buf_sz, max_total * 1e6, tcomm * 1e6, sleep_time);
    }
    
    if(rank == target)
        for(int i = 0; i < buf_sz; ++i) {
            if(winbuf[i] != origin) {
                printf("Error: winbuf[%d] = %d is not %d\n", i, winbuf[i], origin);
            }
        }
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
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