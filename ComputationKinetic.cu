__global__ void GetKinetic_Kernel(AtomInfo* d_atoms ,int N ,double* d_EK ,int *Signal )
{
    __shared__ double per_kinetic[128];
    int id = threadIdx.x +  blockIdx.x * blockDim.x; // this atom's id 
    per_kinetic[threadIdx.x] = 0.0;

     if (id < N)
    {
      per_kinetic[threadIdx.x] = 0.5 * d_atoms[id].mass * (d_atoms[id].velocity[0] * d_atoms[id].velocity[0] + d_atoms[id].velocity[1] * d_atoms[id].velocity[1] + d_atoms[id].velocity[2] * d_atoms[id].velocity[2]);
    }
    __syncthreads();

    if ((id < N) && (threadIdx.x == 0)) // block add to Ek
  {
     int i;
     double Block_kinetic_sum = 0.0;
     for (i = 0; i < blockDim.x; i++)
    {
      Block_kinetic_sum = Block_kinetic_sum + per_kinetic[i];
    }

     while( atomicCAS(Signal,0,1) != 0 ) ;   // lock()
       *d_EK = *d_EK + Block_kinetic_sum ;  // add the Block_kinetic_sum to d_EK
     atomicExch(Signal,0);  // unlock() 
  }

}


// computation the kenetic 
void Computation_Kinetic(AtomInfo **d_atoms ,int* h_ParticleCount, double **d_EK, int **Signal, int deviceCount)
{
   for(int i=0;i<deviceCount;i++)
 {
    int dimgrid;
    dim3 dimBlock = 128;
    int N = h_ParticleCount[i];  // subdomain has atom number

    if (N % 128 == 0)
      dimgrid = N / 128;
    else
      dimgrid = N / 128 + 1;

    dim3 dimGrid = dimgrid;

    cudaSetDevice(i);
    GetKinetic_Kernel<<<dimGrid,dimBlock>>>(d_atoms[i], N, d_EK[i], Signal[i]);
    CUDA_CHECK_ERROR();

 }
  // 计算动能 同步！
  All_devices_Synchronize(deviceCount); 

}



void d_Ek_Set_Zero(double **d_EK, int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
  {
     cudaSetDevice(i);
     CUDA_CALL( cudaMemset(d_EK[i], 0, sizeof(double) ) ); //set d_EK[i] to 0
  }
  
}



double Get_Total_Kinetic(double **h_EK, double **d_EK, int deviceCount)
{
   for(int i=0;i<deviceCount;i++)
  {
    cudaSetDevice(i);
    CUDA_CALL( cudaMemcpy(h_EK[i],d_EK[i] ,sizeof(double) ,cudaMemcpyDeviceToHost) );
  }

  double total_EK = 0.0;
   for(int i=0;i<deviceCount;i++)
  {
     total_EK += h_EK[i][0];
     //printf("\n%dth GPU:%f\t\n",i,h_EK[i][0]);
  }

  return total_EK;
}