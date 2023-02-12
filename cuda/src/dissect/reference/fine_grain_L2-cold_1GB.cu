#include "myutil.hpp"
#include "util.cuh"

//compile nvcc *.cu -o test

__global__ void global_latency (unsigned int * my_array, int array_length, int iterations,  unsigned int * duration, unsigned int *index);


void parametric_measure_global(int N, int iterations, int stride, char *filename);

void measure_global();

void GetFilename(char *filename)
{
    time_t timep;
    struct tm *p;

    time(&timep);          // 获取从1970至今过了多少秒，存入time_t类型的timep
    p = localtime(&timep); // 用localtime将秒数转化为struct tm结构体
    // 把格式化的时间写入字符数组中
    char path[96];
    getcwd(path, sizeof(path));
    // printf("2.2 dir__FILE__: %s\n", dirname(path));
    sprintf(filename, "%s/src/dissect/reference/data/L2_cache_data-%d%d%d.csv",
                dirname(path), p->tm_hour, p->tm_min, p->tm_sec);
}
int main(){

	cudaSetDevice(0);

	measure_global();

	cudaDeviceReset();
	return 0;
}


void measure_global() {

	int N, iterations, stride; 
	//stride in element
	iterations = 1;

    char *filename;
    filename = (char *)malloc(sizeof(char) * 256);
    GetFilename(filename);

	N = 1024 * 1024* 1024/sizeof(unsigned int); //in element
	for (stride = 1; stride <= N/2; stride*=2) {
		printf("\n=====%d GB array, cold cache miss, read 256 element====\n", N/1024/1024/1024);
		printf("Stride = %d element, %d bytes\n", stride, stride * sizeof(unsigned int));
		parametric_measure_global(N, iterations, stride ,filename);
		printf("===============================================\n\n");
	}
    
    free(filename);
}


void parametric_measure_global(int N, int iterations, int stride, char *filename) {
	cudaDeviceReset();
	
	int i;
	unsigned int * h_a;
	/* allocate arrays on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N+2));
	unsigned int * d_a;
	/* allocate arrays on GPU */
	cudaMalloc ((void **) &d_a, sizeof(unsigned int) * (N+2));

   	/* initialize array elements on CPU with pointers into d_a. */	
	for (i = 0; i < N; i++) {
	//original:	
		h_a[i] = (i+stride)%N;
	}

	h_a[N] = 0;
	h_a[N+1] = 0;
	/* copy array elements from CPU to GPU */
    cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

	unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*256);
	unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int)*256);

	unsigned int *duration;
	cudaMalloc ((void **) &duration, sizeof(unsigned int)*256);

	unsigned int *d_index;
	cudaMalloc( (void **) &d_index, sizeof(unsigned int)*256 );

	cudaThreadSynchronize ();
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1,1,1);
	global_latency <<<Dg, Db>>>(d_a, N, iterations,  duration, d_index);
	cudaThreadSynchronize ();

	cudaError_t error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error kernel is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();
    cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int)*256, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int)*256, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize ();\

    // 如果输出文件不存在，则创建文件并写入标题
    if (!isFileExists(filename))
    {
        // 读写文件。文件存在则追加写入，不存在则创建一个新文件
        FILE *fp = fopen(filename, "a+");
        // 如果打开文件失败
        if (fp == NULL)
        {
            // std::cout << "Can't open file : " << filename << std::endl;
            printf("Can't open file : %s \n", filename);
            fprintf(stderr, "fopen() failed.\n");
            exit(EXIT_FAILURE);
        }
        // 标题
        fprintf(fp, "Index,Exec_time,stride\n");
        fclose(fp);
    }

    // 读写文件。文件存在则追加写入，不存在则创建一个新文件
    FILE *fp = fopen(filename, "a+");
    // 如果打开文件失败
    if (fp == NULL)
    {
        // std::cout << "Can't open file : " << filename << std::endl;
        printf("Can't open file : %s \n", filename);
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }

	for(i=0;i<256;i++){
        fprintf(fp, "%d,%d,%d\n", h_index[i], h_timeinfo[i],stride);
        printf("%d\t %d\n", h_index[i], h_timeinfo[i]);
    }
	fclose(fp);	

	/* free memory on GPU */
	cudaFree(d_a);
	cudaFree(d_index);
	cudaFree(duration);

    /*free memory on CPU */
    free(h_a);
    free(h_index);
	free(h_timeinfo);
	
	cudaDeviceReset();
}

__global__ void global_latency (unsigned int * my_array, int array_length, int iterations, unsigned int * duration, unsigned int *index) {

	unsigned int start_time, end_time;
	unsigned int j = 0; 

	__shared__ unsigned int s_tvalue[256];
	__shared__ unsigned int s_index[256];

	int k;

	for(k=0; k<256; k++){
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

    /* for loop 256 times */
	for (k = 0; k < iterations*256; k++) {		
		start_time = clock();
		
		j = my_array[j];
		s_index[k]= j;
		end_time = clock();

        /* record execution time */
		s_tvalue[k] = end_time-start_time;
	}

	my_array[array_length] = j;
	my_array[array_length+1] = my_array[j];

	for(k=0; k<256; k++){
		index[k]= s_index[k];
		duration[k] = s_tvalue[k];
	}
}



