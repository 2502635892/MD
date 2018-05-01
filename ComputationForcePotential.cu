//  ==================force computation kernel================

__global__ void Getforce_Potential_kernel(AtomInfo *d_atoms ,int N ,int *d_C_T ,TableInfo T_Info ,double *d_U ,int *U_lock )
{
  __shared__ double local_d_U[128];

  double f[Dim] = {0.0, 0.0, 0.0};
  double r12[Dim] = {0.0, 0.0, 0.0};
  double d12_square = 0.0;
  double d_6 = 0.0;
  double d_8 = 0.0;
  double d_12 = 0.0;
  double d_14 = 0.0;
  double f12 = 0.0;
  //double sigma_6 = pow(SIGMA, 6);
  //double sigma_12 = sigma_6 * sigma_6;
  int Cell_coordinate[Dim] = {0, 0, 0};
  int cell_id = 0;

  //int i = 0;
  int j = 0;
  int ci = 0;
  int cj = 0;
  int ck = 0;

  int neibor_x = 0;
  int neibor_y = 0;
  int neibor_z = 0;

  double Imag_x = 0.0;
  double Imag_y = 0.0;
  double Imag_z = 0.0;

  double Box_Lx = L_x * a_x;
  double Box_Ly = L_y * a_y;
  double Box_Lz = L_z * a_z; // the simulation area Box size

  double k_x = 0.0;
  double k_y = 0.0;
  double k_z = 0.0;

  int id = threadIdx.x + blockIdx.x * blockDim.x;  // this atom's id
  local_d_U[threadIdx.x] = 0.0;                   // location potential energy

  if (id < N)
  {

    Cell_coordinate[0] = (int)(d_atoms[id].position[0] / T_Info.L_cell[0]); // locatied the idth atom in which cell
    Cell_coordinate[1] = (int)(d_atoms[id].position[1] / T_Info.L_cell[1]);
    Cell_coordinate[2] = (int)(d_atoms[id].position[2] / T_Info.L_cell[2]); //buge two how to map int the right cell

    f[0] = 0.0;
    f[1] = 0.0;
    f[2] = 0.0;

    for (ci = Cell_coordinate[0] - 1; ci <= Cell_coordinate[0] + 1; ci++)
    {
      for (cj = Cell_coordinate[1] - 1; cj <= Cell_coordinate[1] + 1; cj++)
      {
        for (ck = Cell_coordinate[2] - 1; ck <= Cell_coordinate[2] + 1; ck++)
        {
          neibor_x = (ci + T_Info.cell_num[0]) % T_Info.cell_num[0];
          neibor_y = (cj + T_Info.cell_num[1]) % T_Info.cell_num[1];
          neibor_z = (ck + T_Info.cell_num[2]) % T_Info.cell_num[2]; 
                                                        //perodical bound condition(PBC)
          cell_id = neibor_x + neibor_y * T_Info.cell_num[0] + neibor_z * T_Info.cell_num[0] * T_Info.cell_num[1]; //get neibor cell id

          for (j = 0; j < d_C_T[cell_id * Max_Mem + Max_Mem - 1]; j++)
          {
            if (d_C_T[cell_id * Max_Mem + j] != id)
            {

              k_x = floor((double)(ci) / (double)T_Info.cell_num[0]);

              k_y = floor((double)(cj) / (double)T_Info.cell_num[1]);

              k_z = floor((double)(ck) / (double)T_Info.cell_num[2]);

              Imag_x = k_x * Box_Lx + d_atoms[d_C_T[cell_id * Max_Mem + j]].position[0];
              Imag_y = k_y * Box_Ly + d_atoms[d_C_T[cell_id * Max_Mem + j]].position[1];
              Imag_z = k_z * Box_Lz + d_atoms[d_C_T[cell_id * Max_Mem + j]].position[2]; // pbc

              r12[0] = Imag_x - d_atoms[id].position[0];
              r12[1] = Imag_y - d_atoms[id].position[1];
              r12[2] = Imag_z - d_atoms[id].position[2];

              d12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];

              d_6 = d12_square * d12_square * d12_square;
              d_8 = d12_square * d_6;
              d_12 = d_6 * d_6;
              d_14 = d_6 * d_8;

              f12 = (sigma_6 / d_8 - 2.0 * sigma_12 / d_14) * 24.0 * EPSILON; // LJ coeffiencient for force

              f[0] = f[0] + f12 * r12[0];
              f[1] = f[1] + f12 * r12[1];
              f[2] = f[2] + f12 * r12[2];

              local_d_U[threadIdx.x] = local_d_U[threadIdx.x] + 4.0 * EPSILON * (sigma_12 / d_12 - sigma_6 / d_6);
            }
          }
        }
      }
    }

    d_atoms[id].force[0] = f[0];
    d_atoms[id].force[1] = f[1];
    d_atoms[id].force[2] = f[2];

  } // the Force of atom culculation

  __syncthreads();

  if ((id < N) && (threadIdx.x == 0))
  {
    int i;
    double block_U_sum = 0.0;
    for (i = 0; i < blockDim.x; i++)
    {
      block_U_sum = block_U_sum + local_d_U[i];
    }

    while (atomicCAS(U_lock, 0, 1) != 0)
      ; // lock()

    *d_U = *d_U + block_U_sum;

     atomicExch(U_lock, 0); // unlock()
  }

} // get the d_atoms[id] force and energy !!!



// force computation !!!   
void Computation_Force_Potential(AtomInfo **d_atoms ,int* h_ParticleCount ,int **GlobalCellTable , TableInfo T_Info ,double **d_U , int **U_lock , int deviceCount)
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
      Getforce_Potential_kernel<<<dimGrid,dimBlock>>>(d_atoms[i] , N , GlobalCellTable[i], T_Info, d_U[i], U_lock[i] );
      CUDA_CHECK_ERROR();
    }
    //设备执行同步
    All_devices_Synchronize(deviceCount); 
}



void d_U_Set_Zero(double **d_U , int deviceCount)
{
   for(int i=0;i<deviceCount;i++)
  {
     cudaSetDevice(i);
     CUDA_CALL( cudaMemset(d_U[i], 0, sizeof(double) ) ); //set U_lock[i] to 0
  }
  
}


double Get_Total_Potential(double **h_U, double **d_U, int deviceCount)
{
     for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemcpy(h_U[i],d_U[i] ,sizeof(double) ,cudaMemcpyDeviceToHost) );
    }

    double h_total_U = 0.0;
    
     for(int i=0;i<deviceCount;i++)
    {
      h_total_U += h_U[i][0];
    }

    return h_total_U;
}