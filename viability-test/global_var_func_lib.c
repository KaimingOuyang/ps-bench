#include <stdio.h>
#include <pip.h>

pip_barrier_t gbarrier;

typedef struct fi_cq{
    void (*func)(int, int);
    int rank;
} fi_cq_t;

/* same program, function pointer */
void print_var(fi_cq_t *cq, int caller_rank){
    cq->func(cq->rank, caller_rank);
    return;
}

extern fi_cq_t cq_obj;

void print_rank(int rank, int caller_rank){
    printf("I am caller rank %d, help print rank %d\n", caller_rank, rank);
    fflush(stdout);
    return;
}

int main(){
    int rank, ntasks;
    fi_cq_t *cq_obj_local;
    pip_barrier_t *lbarrier;
	
    pip_init( &rank, &ntasks, NULL, 0 );
	pip_barrier_init(&gbarrier, ntasks);
    pip_get_addr(0, "cq_obj", (void**) &cq_obj_local);
    pip_get_addr(0, "gbarrier", (void**) &lbarrier);
	printf("rank %d - pid %d, cq_obj %p, cq_obj_local %p\n", rank, getpid(), &cq_obj, cq_obj_local);

    if(rank == 0){
        cq_obj.func = &print_rank;
        cq_obj.rank = rank;
		pip_barrier_wait(lbarrier);
    }else{
		pip_barrier_wait(lbarrier);
		print_var(cq_obj_local, rank);
	}

    return 0;
}
