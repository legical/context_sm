#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
int main(int argc,char *argv[])
{
  printf("1. _pgmptr: %s\n",_pgmptr);

  char path[64]; //PATH_MAX is defined in limits.h
  getcwd(path,sizeof(path));
  printf("2. getcwd: %s\n",path);

  printf("3. __FILE__: %s\n",__FILE__);

  printf("4. argv[0]: %s\n",argv[0]);
//   system("pause");
  return 0;
}