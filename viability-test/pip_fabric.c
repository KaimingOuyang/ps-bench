#include <stdio.h>
#include <pip.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

int a = 0;
int global_rank;
void print_var(int caller_rank, int caller_b){
    static int b = 0;

    if(caller_rank == global_rank){
        b = caller_b;
    }

    printf("I am rank %d - help rank %d to call print_var, a = %d, b = %d\n", caller_rank, global_rank, a, b);
}

typedef struct Comm{
    
} Comm_t;

void init_libfabric(){
    hints = fi_allocinfo();
}

int main(){
    int rank, ntasks;
    pip_init( &rank, &ntasks, NULL, 0 );
    init_libfabric();


    return 0;
}