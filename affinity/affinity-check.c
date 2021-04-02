
#define _GNU_SOURCE
#include <stdio.h>
#include <mpi.h>
#include <sched.h>
int main(){
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cpu_set_t mask;
    CPU_ZERO(&mask);
    sched_getaffinity(getpid(), sizeof(cpu_set_t), &mask);
    
    int num_cpu;
    CPU_COUNT_S(num_cpu, &mask);
    for(int i = 0; i < num_cpu; ++i){
        if(CPU_ISSET(i, &mask)){
            printf("rank %d pid %d bind to cpu %d\n", rank, getpid(), i);
        }
    }

    MPI_Finalize();
    return 0;
}