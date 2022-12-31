#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
int main(int argc, char *argv[])
{
    // printf("1. _pgmptr: %s\n", _pgmptr);

    char path[64];
    getcwd(path, sizeof(path));
    printf("1. getcwd: %s\n", path);

    printf("2. __FILE__: %s\n", __FILE__);

    printf("3. argv[0]: %s\n", argv[0]);
    
    return 0;
}