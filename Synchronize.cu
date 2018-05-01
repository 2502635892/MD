void  All_devices_Synchronize(int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
   {
     cudaSetDevice(i);
     CUDA_CALL(cudaDeviceSynchronize());
   } 
}



__global__ void test_atomticADD_kernel(int *d_S,int N)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < N)
  {
    int temp = atomicAdd(d_S,1);
    printf("%dthead: %d\n",id,temp);
  }

}

void test_atomticADD()
{
  dim3 testBlock = 64;
  dim3 testGrid = 2;
  int N =  64 * 2;
  cudaSetDevice(1);

  int *d_S;
  int *h_S = (int*)malloc(sizeof(int));
  CUDA_CALL( cudaMalloc( &d_S, sizeof(int) ) );
  CUDA_CALL( cudaMemset(d_S , 0 , sizeof(int) ) );

  test_atomticADD_kernel<<<testGrid,testBlock>>>(d_S, N);
  CUDA_CHECK_ERROR();

  CUDA_CALL( cudaMemcpy(h_S,d_S ,sizeof(int) ,cudaMemcpyDeviceToHost) );
  printf("%d\n",*h_S);

}


__global__ void test_cudaMalloc_kernel(int *d_p ,int N)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id==0)
  {
    for(int i=0;i<N;i++)
      printf("%d %d %d %d %d %d\n", d_p[i] , *(d_p + i) , &d_p[i] , (d_p + i) , &(*(d_p + i)) , d_p );
  }

}



void test_cudaMalloc()
{
  int *d_p;
  int N = 10;

  cudaSetDevice(0);
  int *p = (int*)malloc(sizeof(int)*N);
  for(int i=0;i<N;i++)
     p[i] = i; 
  
  CUDA_CALL( cudaMalloc(&d_p,sizeof(int)*N) );
  CUDA_CALL( cudaMemcpy(d_p , p ,sizeof(int) * N ,cudaMemcpyHostToDevice) );

  dim3 testBlock = 64;
  dim3 testGrid = 1;
  test_cudaMalloc_kernel<<<testGrid,testBlock>>>(d_p,N);


}
