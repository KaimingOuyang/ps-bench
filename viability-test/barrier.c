#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char *argv[]){
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        sleep(10);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank %d - complete barrier\n", rank);
    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}