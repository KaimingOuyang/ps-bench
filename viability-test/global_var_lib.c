#include <stdio.h>
#include <pip.h>

extern int a;
extern int global_rank;
pip_barrier_t gbarrier;

/* same program, global var */
void print_var(int caller_rank){
    if(caller_rank == global_rank){
        printf("rank %d - original &a = %p\n", global_rank, &a);
    }else{
        printf("rank %d - help rank %d to call print_var, a = %d, &a = %p\n", caller_rank, global_rank, a, &a);
    }
    fflush(stdout);
    return;
}

int main(){
    int rank, ntasks;
    pip_init( &rank, &ntasks, NULL, 0 );
    // printf("rank %d - total have ntask %d\n", rank, ntasks);
    void (*help_print)(int);
	pip_barrier_t *lbarrier;
	global_rank = rank;
	pip_barrier_init(&gbarrier, ntasks);
    pip_get_addr(0, "print_var", (void**) &help_print);
    pip_get_addr(0, "gbarrier", (void**) &lbarrier);
	printf("rank %d - pid %d, print_var %p, help_print %p\n", rank, getpid(), print_var, help_print);

    if(rank == 0){
        a = 2;
        print_var(rank);
		pip_barrier_wait(lbarrier);
    }else{
		pip_barrier_wait(lbarrier);
		help_print(rank);
	}
    return 0;
}
