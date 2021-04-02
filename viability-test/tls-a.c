#include <stdio.h>
#include <pip.h> /* new include path */

#define N       (10)

__thread int x[N], a, b, c;

void print_vars( int pipid ) {
  printf( "A[%d]: a=%d  b=%d  c=%d\n", pipid, a, b, c );
}

int main() {
  int i, pipid;
  void *func;

  for( i=0; i<N; i++ ) x[i] = i * 100;
  a = -123;
  b = 456;
  c = 999;
  /* no need of calling pip_init() explicitly in PiP-v2 or higher */
  pip_get_pipid( &pipid );
  print_vars( pipid );
  pip_named_export( (void*) print_vars, "X" ); /* new func */
  pip_named_import( pipid^1, (void**)&func, "X" ); /* new func */
  ((void(*)(int))func)( pipid );
  return 0;
}

