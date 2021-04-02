#include <stdio.h>
#include <pip.h>

int a = 0;
int global_rank;
pip_barrier_t gbarrier;

/* same program, global var */
void print_var(int caller_rank, int caller_b){
    static int b = 0;

    if(caller_rank == global_rank){
        b = caller_b;
        printf("rank %d - original &a = %p, &b = %p\n", global_rank, &a, &b);
        fflush(stdout);
    }else{
        printf("rank %d - help rank %d to call print_var, a = %d, b = %d, &a = %p, &b = %p\n", caller_rank, global_rank, a, b, &a, &b);
    }
    return;
}


/* same program, function pointer */

uint64_t func_addr;

int main(){
    int rank, ntasks;
    pip_init( &rank, &ntasks, NULL, 0 );
    // printf("rank %d - total have ntask %d\n", rank, ntasks);
    void (*help_print)(int, int);
	pip_barrier_t *lbarrier;
	global_rank = rank;
	pip_barrier_init(&gbarrier, ntasks);
    pip_get_addr(0, "print_var", (void**) &help_print);
    pip_get_addr(0, "gbarrier", (void**) &lbarrier);
	printf("rank %d - pid %d, print_var %p, help_print %p\n", rank, getpid(), print_var, help_print);

    if(rank == 0){
        a = 2;
        print_var(rank, 200);
		pip_barrier_wait(lbarrier);
    }else{
		pip_barrier_wait(lbarrier);
		help_print(rank, 100);
	}
    return 0;
}
