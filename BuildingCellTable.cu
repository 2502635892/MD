void GlobalCellTable_Set_Zero(int **GlobalCellTable,TableInfo T_Info,int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
  {
     cudaSetDevice(i);
     CUDA_CALL( cudaMemset( GlobalCellTable[i] , 0 , sizeof(int) * Max_Mem * T_Info.Total_cellNUM ) );  // initialize element = 0
  }  // cudaMemset 同步调用
 
}




// mapping the atom to local cell table kernel
__global__ void MapCellTable_Kernel(AtomInfo *d_atoms, int N, int *d_C_T, TableInfo T_Info)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    //int x,y,z;  // atom position

    double x,y,z;

    int Cell_coordinate_x;
    int Cell_coordinate_y;
    int Cell_coordinate_z;

    int cell_id;
    int count;

     if( id < N )
    {
       x = d_atoms[id].position[0];
       y = d_atoms[id].position[1];
       z = d_atoms[id].position[2]; // get the atom position

       Cell_coordinate_x = (int)(x / T_Info.L_cell[0]);
       Cell_coordinate_y = (int)(y / T_Info.L_cell[1]);
       Cell_coordinate_z = (int)(z / T_Info.L_cell[2]);   // get the  idth atom  coordinate

       cell_id = Cell_coordinate_x + Cell_coordinate_y * T_Info.cell_num[0] + Cell_coordinate_z * T_Info.cell_num[0] * T_Info.cell_num[1];

       count = atomicAdd( &d_C_T[Max_Mem * cell_id  + Max_Mem - 1] , 1);  // get the index of the  cell_id's th cell!!!

       d_C_T[Max_Mem * cell_id  + count ] = id;   // mapping the id th atom to the cell_id's th cell!
  
    }


}



void Mapping_to_CellTable(AtomInfo **d_atoms , int **GlobalCellTable , TableInfo T_Info , int *h_AddShell_ParticleCount , int deviceCount)
{
     for(int i=0;i<deviceCount;i++)
    {
      int dimgrid;
      dim3 dimBlock = 128;
      int N =  h_AddShell_ParticleCount[i];  // subdomain has atom number(including shell cell atom)
      if (N % 128 == 0)
        dimgrid = N / 128;
      else
        dimgrid = N / 128 + 1;

      dim3 dimGrid = dimgrid;

      cudaSetDevice(i);
      MapCellTable_Kernel<<<dimGrid,dimBlock>>>( d_atoms[i], N, GlobalCellTable[i] , T_Info );
      CUDA_CHECK_ERROR();
    }   
  
    All_devices_Synchronize(deviceCount); 
}


