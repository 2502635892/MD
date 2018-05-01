__global__ void UpdateAtomsInfo_kernel_1(AtomInfo *d_atoms, int N, double dt)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x; //this atom's id

  //the simulation area Box size
  double Box_Lx = L_x * a_x;
  double Box_Ly = L_y * a_y;
  double Box_Lz = L_z * a_z;  

  dt = dt / TIME_UNIT_CONVERSION;

  if (id < N)
  {
    d_atoms[id].velocity[0] = d_atoms[id].velocity[0] + d_atoms[id].force[0] / d_atoms[id].mass * dt * 0.5;
    d_atoms[id].velocity[1] = d_atoms[id].velocity[1] + d_atoms[id].force[1] / d_atoms[id].mass * dt * 0.5;
    d_atoms[id].velocity[2] = d_atoms[id].velocity[2] + d_atoms[id].force[2] / d_atoms[id].mass * dt * 0.5;

    d_atoms[id].position[0] = d_atoms[id].position[0] + d_atoms[id].velocity[0] * dt;
    d_atoms[id].position[1] = d_atoms[id].position[1] + d_atoms[id].velocity[1] * dt;
    d_atoms[id].position[2] = d_atoms[id].position[2] + d_atoms[id].velocity[2] * dt; 
    
    // pbc operation
    d_atoms[id].position[0] = d_atoms[id].position[0] - Box_Lx * floor(d_atoms[id].position[0] / Box_Lx);
    d_atoms[id].position[1] = d_atoms[id].position[1] - Box_Ly * floor(d_atoms[id].position[1] / Box_Ly);
    d_atoms[id].position[2] = d_atoms[id].position[2] - Box_Lz * floor(d_atoms[id].position[2] / Box_Lz); //PBC
  }

}




void Update_AtomsInfo_kernel_1(AtomInfo **d_atoms , int* h_ParticleCount , double dt,int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
  {
    int dimgrid;
    dim3 dimBlock = 128;
    int N =  h_ParticleCount[i];   // subdomain has atom number
    if (N % 128 == 0)
      dimgrid = N / 128;
    else
      dimgrid = N / 128 + 1;

    dim3 dimGrid = dimgrid;

    cudaSetDevice(i);
    UpdateAtomsInfo_kernel_1<<<dimGrid,dimBlock>>>(d_atoms[i], N, dt);
    CUDA_CHECK_ERROR();

   }
    // 更新 执行同步
   All_devices_Synchronize(deviceCount);

}






__global__ void test_UpdateAtomsInfo_kernel_1(AtomInfo *d_atoms, int N ,TableInfo T_Info ,deProcess P , int sudDomainId)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;  //this atom's id

  if( id == 0 )
  {
      int cx,cy,cz;
      int subdomain_id;
      int ecape_count = 0 ;

      for(int i=0;i<N;i++)
    {
         if(i==0 || i==(N-1) )
        {
            printf("%dth: P: %f %f %f \t  V:%f %f %f \t F:%f %f %f\n",i,d_atoms[i].position[0],d_atoms[i].position[1],d_atoms[i].position[2],
            d_atoms[i].velocity[0],d_atoms[i].velocity[1],d_atoms[i].velocity[2], 
            d_atoms[i].force[0],d_atoms[i].force[1],d_atoms[i].force[2]);
        }

        cx = (int)(d_atoms[i].position[0] / T_Info.L_cell[0]);
        cy = (int)(d_atoms[i].position[1] / T_Info.L_cell[1]);
        cz = (int)(d_atoms[i].position[2] / T_Info.L_cell[2]);

        subdomain_id = d_cellToSubDomainId(cx , cy , cz , T_Info , P );

        if(subdomain_id != sudDomainId )
            ecape_count ++ ; 
     

    }

    printf("the %dth gpu check escape number is %d\n ",sudDomainId,ecape_count);

  }


}




void test_Update_AtomsInfo_kernel_1(AtomInfo **d_atoms , int* h_ParticleCount ,TableInfo T_Info ,int deviceCount,deProcess P )
{
  
     for(int i=0;i<deviceCount;i++) 
    { 
      dim3 testBlock = 128;
      dim3 testGrid = 1;
      int N = h_ParticleCount[i];
      cudaSetDevice(i);
      test_UpdateAtomsInfo_kernel_1<<<testGrid,testBlock>>>(d_atoms[i],  N , T_Info , P , i);
      CUDA_CHECK_ERROR();
    }

}






__global__ void UpdateAtomsInfo_kernel_2(AtomInfo *d_atoms, int N, double dt)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x; //this atom's id

  dt = dt / TIME_UNIT_CONVERSION;

  if (id < N)
  {
    d_atoms[id].velocity[0] = d_atoms[id].velocity[0] + d_atoms[id].force[0] / d_atoms[id].mass * dt * 0.5;
    d_atoms[id].velocity[1] = d_atoms[id].velocity[1] + d_atoms[id].force[1] / d_atoms[id].mass * dt * 0.5;
    d_atoms[id].velocity[2] = d_atoms[id].velocity[2] + d_atoms[id].force[2] / d_atoms[id].mass * dt * 0.5;
  }
  
}

void Update_AtomsInfo_kernel_2(AtomInfo **d_atoms , int* h_ParticleCount ,double dt,int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
  {
    int dimgrid;
    dim3 dimBlock = 128;
    int N =  h_ParticleCount[i];  // subdomain has atom number
    if (N % 128 == 0)
      dimgrid = N / 128;
    else
      dimgrid = N / 128 + 1;

    dim3 dimGrid = dimgrid;

    cudaSetDevice(i);
    UpdateAtomsInfo_kernel_2<<<dimGrid,dimBlock>>>(d_atoms[i], N, dt);
    CUDA_CHECK_ERROR();

   }

   All_devices_Synchronize(deviceCount);

}








__global__ void Scale_Velocity_kernel(AtomInfo* d_atoms , int N , double double_ek , int Total_Particle )
{
   double scalar; 
   int id = threadIdx.x +  blockIdx.x * blockDim.x; // this atom's id  
   if( id < N )
   {
      scalar = sqrt( Termp * Dim * K_B * Total_Particle / double_ek); // scale coefficient 

      d_atoms[id].velocity[0] = d_atoms[id].velocity[0] * scalar;
      d_atoms[id].velocity[1] = d_atoms[id].velocity[1] * scalar;
      d_atoms[id].velocity[2] = d_atoms[id].velocity[2] * scalar;
   }
       
}



void Scale_Velocity(AtomInfo **d_atoms ,int* h_ParticleCount , double double_ek , int deviceCount , int Total_Particle)
{
    for(int i=0;i<deviceCount;i++)
  {
    int dimgrid;
    dim3 dimBlock = 128;
    int N =  h_ParticleCount[i];  // subdomain has atom number
    if (N % 128 == 0)
      dimgrid = N / 128;
    else
      dimgrid = N / 128 + 1;

    dim3 dimGrid = dimgrid;

    cudaSetDevice(i);
    Scale_Velocity_kernel<<<dimGrid,dimBlock>>>(d_atoms[i] , N , double_ek ,Total_Particle);
    CUDA_CHECK_ERROR();

   }
   
   All_devices_Synchronize(deviceCount);

}




















