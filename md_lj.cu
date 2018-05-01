#include "def.h"
#include "dataStruct.h"
#include "initial.c"
#include "Synchronize.cu"
#include "DomainDecomp.cu"
#include "BuildingCellShell.cu"
#include "ParticlePartition.cu"
#include "BuildingCellTable.cu"
#include "ComputationForcePotential.cu"
#include "VerletIntegration.cu"
#include "ComputationKinetic.cu"
#include "EscapeAdjust.cu"
#include "Check.cu"

void md_lj(void)
{
  int n0 = FCC_NUM;
  int nxyz[3] = {L_x, L_y, L_z};
  double a[3] = {a_x, a_y, a_z};
  double T = Termp;
  int N = n0 * nxyz[0] * nxyz[1] * nxyz[2];  // total atom number in the system 
  double dt = DT;   // integration step == DT==5
  deProcess P;      // process x,y,z
  TableInfo T_Info; // table info
  // process decompoment
  P.x = 2;
  P.y = 1;
  P.z = 1;

  int Pnum = P.x * P.y * P.z; //  Pnum : process numbers !

  AtomInfo *atoms = (AtomInfo *)malloc(sizeof(AtomInfo) * N);  // malloc the memory for the total atom 
  initialize(atoms, N, nxyz, a, T);   //initialize the atoms information
  GetTableInfo(T_Info);               // get the cell divided information

  // --------- segment malloc memeory-----------
  seg *Xseg = (seg *)malloc(sizeof(seg) * P.x);
  seg *Yseg = (seg *)malloc(sizeof(seg) * P.y);
  seg *Zseg = (seg *)malloc(sizeof(seg) * P.z);
  
  // ---------segment split three coordinates ------------
  segPartiton(Xseg ,P.x ,T_Info.cell_num[0]);
  segPartiton(Yseg ,P.y ,T_Info.cell_num[1]);
  segPartiton(Zseg ,P.z ,T_Info.cell_num[2]);

  printf("%d %d %d\n",T_Info.cell_num[0],T_Info.cell_num[1],T_Info.cell_num[2]);
  
  //==++++++++++++++++++++ subdomain seg info array++++++++++++++++++++++++
  subDomain *subdomain = (subDomain *)malloc(sizeof(subDomain) * Pnum);  //  subdomain seg info
  
  int index = 0;
    for(int i=0; i<P.z; i++)
  { 
      for(int j=0;j<P.y;j++)
    {
         for(int k=0;k<P.x;k++)
       {
           subdomain[index].X = Xseg[k];
           subdomain[index].Y = Yseg[j];
           subdomain[index].Z = Zseg[i];
           printf("%d subdom:X(%d,%d)\n",index,subdomain[index].X.low,subdomain[index].X.high);
           printf("%d subdom:Y(%d,%d)\n",index,subdomain[index].Y.low,subdomain[index].Y.high);
           printf("%d subdom:Z(%d,%d)\n",index,subdomain[index].Z.low,subdomain[index].Z.high);
		       index ++;
       }
       
	}
  
  } // subdomain partition 
  


// malloc the memory for the subdomain atomInfo array  on CPU
int ATOMINFO_Malloc_Len ;
ATOMINFO_Malloc_Len = N/Pnum * SCALER ;

AtomInfo **atomsInfoArr = (AtomInfo **)malloc(sizeof(AtomInfo*) * Pnum);   // atomsInfoArr
for(int i=0;i<Pnum;i++)
  atomsInfoArr[i] = (AtomInfo *)malloc( sizeof(AtomInfo)  *  ATOMINFO_Malloc_Len );

// malloc  for subdomain atom number counter
int* h_ParticleCount = (int*)malloc( sizeof(int) * Pnum );
memset(h_ParticleCount,0,sizeof(int)*Pnum); // atomInfo array continue number

int* h_AddShell_ParticleCount = (int*)malloc( sizeof(int) * Pnum );
memset(h_AddShell_ParticleCount,0,sizeof(int)*Pnum); // atomInfo array continue number


MapParticleToSubdomain(atoms,atomsInfoArr,ATOMINFO_Malloc_Len,T_Info, P, h_ParticleCount, N, Pnum);

for(int i=0;i<Pnum;i++)
   printf("%dth:%d\n",i,h_ParticleCount[i]);



// get the boundary cell number of the sub domain!
int * SubDomainBoundarySize = (int *)malloc(sizeof(int) * Pnum);
BocellNumSubdomain(SubDomainBoundarySize,subdomain,Pnum); 

// cell shell size decision the SendBuffer & receiveBuffer size 
int max_subdomainshell = 0;
for(int i=0;i<Pnum;i++)
{
 if (SubDomainBoundarySize[i] > max_subdomainshell )
  max_subdomainshell = SubDomainBoundarySize[i];
}

int STEP = max_subdomainshell * rh / 2; 

int RECEVIE_Malloc_Len ;
    RECEVIE_Malloc_Len = STEP * Pnum ;

int SEND_Malloc_Len ;
    SEND_Malloc_Len =  STEP * Pnum ;

 // -------------Get the GPUs count number--------------------
 int deviceCount;
 CUDA_CALL( cudaGetDeviceCount(&deviceCount) );
 printf("total GPU number is  %d  in system\n",deviceCount);

 deviceCount = Pnum;

 
 // Reset device
 for(int i=0;i<deviceCount;i++)
  {
    cudaSetDevice(i);
    cudaDeviceReset();
  } 

  // malloc memory on GPU remember Free
  AtomInfo **d_atoms = (AtomInfo **)malloc(sizeof(AtomInfo*) * deviceCount) ;
   for(int i=0;i<deviceCount;i++)
  {
    cudaSetDevice(i);
    CUDA_CALL( cudaMalloc(&d_atoms[i], sizeof(AtomInfo) * ATOMINFO_Malloc_Len) ); 
  }

  // copy atomInfo to GPU memory from CPU 
   for(int i=0;i<deviceCount;i++)
  {
     cudaSetDevice(i);
     CUDA_CALL( cudaMemcpy(d_atoms[i], atomsInfoArr[i], sizeof(AtomInfo) * h_ParticleCount[i], cudaMemcpyHostToDevice) );
  }



AtomInfo **SendBuffer = (AtomInfo **)malloc(sizeof(AtomInfo*) * deviceCount) ;
 for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);                       // SEND_Malloc_Len =  STEP * Pnum
  CUDA_CALL( cudaMalloc( &SendBuffer[i] , sizeof(AtomInfo) * SEND_Malloc_Len ) ); 
}

AtomInfo **ReceBuffer = (AtomInfo **)malloc(sizeof(AtomInfo*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
   cudaSetDevice(i);                     // RECEVIE_Malloc_Len = STEP * Pnum
   CUDA_CALL( cudaMalloc(&ReceBuffer[i], sizeof(AtomInfo) * RECEVIE_Malloc_Len ) );  
}

int **d_BuildShell_SendCount = (int **)malloc( sizeof(int*) * deviceCount );
  for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);
  CUDA_CALL( cudaMalloc(&d_BuildShell_SendCount[i], sizeof(int)*deviceCount) );    
}

int ** h_BuildShell_SendCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  h_BuildShell_SendCount[i] = (int *)malloc( sizeof(int)*deviceCount );
}

int ** d_BuildShell_ReceCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);
  CUDA_CALL( cudaMalloc(&d_BuildShell_ReceCount[i], sizeof(int)*deviceCount) );  
}

int ** h_BuildShell_ReceCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  h_BuildShell_ReceCount[i] = (int *)malloc( sizeof(int)*deviceCount );
}

int **GlobalCellTable = (int **)malloc( sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
   cudaSetDevice(i);
   CUDA_CALL( cudaMalloc( &GlobalCellTable[i], sizeof(int) * Max_Mem * T_Info.Total_cellNUM ) ); 
} 



// computation Force and potencial energy
double **d_U = (double **)malloc(sizeof(double*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
 {
   cudaSetDevice(i);
   CUDA_CALL( cudaMalloc(&d_U[i], sizeof(double) ) );
 }

 int **U_lock = (int **)malloc(sizeof(int*) * deviceCount);
 for(int i=0;i<deviceCount;i++)
 {
   cudaSetDevice(i);
   CUDA_CALL( cudaMalloc(&U_lock[i], sizeof(int) ) );
   CUDA_CALL( cudaMemset(U_lock[i], 0, sizeof(int) ) ); //set U_lock[i] to 0
 }

 double **h_U = (double **)malloc(sizeof(double*) * deviceCount);
 for(int i=0;i<deviceCount;i++)
 {
   h_U[i] = (double *)malloc( sizeof(double) );
 }




// malloc the d_EK for the kenetic computation
double **d_EK = (double **)malloc(sizeof(double*) * deviceCount);
for(int i=0;i<deviceCount;i++)
{
 cudaSetDevice(i);
 CUDA_CALL( cudaMalloc(&d_EK[i], sizeof(double) ) );
}

int **Signal = (int **)malloc( sizeof(int*) * deviceCount);
for(int i=0;i<deviceCount;i++)
{
 cudaSetDevice(i);
 CUDA_CALL( cudaMalloc( &Signal[i], sizeof(int) ) );
 CUDA_CALL( cudaMemset(Signal[i], 0 , sizeof(int) ) ); //set d_EK[i] to 0
}

double **h_EK = (double **)malloc(sizeof(double*) * deviceCount);
 for(int i=0;i<deviceCount;i++)
{
 h_EK[i] = (double *)malloc( sizeof(double) );
}




// adjust array malloc on gpus
AtomInfo **d_Adjust = (AtomInfo **)malloc(sizeof(AtomInfo*) * deviceCount) ;
for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);
  CUDA_CALL( cudaMalloc(&d_Adjust[i], sizeof(AtomInfo) * ATOMINFO_Malloc_Len) ); // malloc memory on GPU remember Free
}


// Escape particle adjustment vars malloc
int **d_LingerCount = (int **)malloc(sizeof(int*) * deviceCount);
for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);
  CUDA_CALL( cudaMalloc( &d_LingerCount[i] , sizeof(int) ) ); // malloc memory on GPU remember Free
}

int **h_LingerCount = (int **)malloc( sizeof(int*) * deviceCount);
for(int i=0;i<deviceCount;i++)
{
  h_LingerCount[i] = (int*)malloc( sizeof(int) );
}


int **d_Escape_SendCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);
  CUDA_CALL( cudaMalloc(&d_Escape_SendCount[i], sizeof(int)*deviceCount) );    
}

int ** h_Escape_SendCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  h_Escape_SendCount[i] = (int *)malloc( sizeof(int)*deviceCount );
}

int ** d_Escape_ReceCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  cudaSetDevice(i);
  CUDA_CALL( cudaMalloc(&d_Escape_ReceCount[i], sizeof(int)*deviceCount) );  
}

int ** h_Escape_ReceCount = (int **)malloc(sizeof(int*) * deviceCount);
  for(int i=0;i<deviceCount;i++)
{
  h_Escape_ReceCount[i] = (int *)malloc( sizeof(int)*deviceCount );
}


double total_U = 0.0;
double total_Ek = 0.0;





// Mapping Cell Table on GPUS
d_BuildShell_SendCount_Set_Zero(d_BuildShell_SendCount , deviceCount );
Load_Shell_Atoms_To_SendBuffer(d_atoms , h_ParticleCount ,SendBuffer ,deviceCount ,P ,T_Info ,Pnum ,d_BuildShell_SendCount ,STEP );


d_BuildShell_SendCount_DeviceToHost(d_BuildShell_SendCount, h_BuildShell_SendCount, deviceCount);
Build_Shell_ReceiveBuffer_From_SendBuffer(ReceBuffer , SendBuffer , h_BuildShell_SendCount , deviceCount , STEP );


Build_Shell_ReceiveCount_From_SendCount( d_BuildShell_ReceCount, d_BuildShell_SendCount , deviceCount);
Build_Shell_ReceCount_DeviceToHost( h_BuildShell_ReceCount ,  d_BuildShell_ReceCount , deviceCount) ;
Build_Shell_Insert_ReceBuffer_Fellow_AtomInfoArray(d_atoms , ReceBuffer , h_BuildShell_ReceCount , h_ParticleCount , h_AddShell_ParticleCount , STEP , deviceCount );

GlobalCellTable_Set_Zero( GlobalCellTable , T_Info , deviceCount );
Mapping_to_CellTable( d_atoms , GlobalCellTable , T_Info , h_AddShell_ParticleCount , deviceCount);



// Computation Force and Potential on GPUs
 
 d_U_Set_Zero( d_U , deviceCount);
 Computation_Force_Potential(d_atoms , h_ParticleCount ,GlobalCellTable , T_Info , d_U , U_lock , deviceCount);
 
 
 total_U = Get_Total_Potential(h_U, d_U, deviceCount);

 // Get the Kinetic on GPUs
 d_Ek_Set_Zero(d_EK , deviceCount);
 Computation_Kinetic(d_atoms , h_ParticleCount, d_EK, Signal, deviceCount);
 total_Ek =  Get_Total_Kinetic(h_EK, d_EK, deviceCount);
 printf("kinetic is %f \t Potential is %f \n", total_Ek/N , 0.5 * total_U / N );



 int Ne = NE;
 int Np = NP;


   for(int step=1;step<=Ne+Np;step++)
 {
 
  
 // Update partition of Velocity and total Position
   Update_AtomsInfo_kernel_1(d_atoms , h_ParticleCount , dt , deviceCount);



// Escape particle Adjustment
d_LingerCount_Set_Zero( d_LingerCount, deviceCount);
d_Escape_SendCount_Set_Zero( d_Escape_SendCount , deviceCount);
Split_Escape_And_linger( d_atoms , h_ParticleCount , d_Adjust, d_LingerCount , SendBuffer, d_Escape_SendCount , deviceCount, P, T_Info , Pnum , STEP);


LingerCount_DeviceToHost(h_LingerCount , d_LingerCount, deviceCount);
linger_To_AtomInfoArray( d_atoms , d_Adjust , h_LingerCount , deviceCount);

d_Escape_SendCount_DeviceToHost(d_Escape_SendCount , h_Escape_SendCount , deviceCount);

d_Escape_ReceiveCount_From_SendCount(d_Escape_ReceCount,d_Escape_SendCount , deviceCount);
d_Escape_ReceCount_DeviceToHost(h_Escape_ReceCount , d_Escape_ReceCount , deviceCount);


Escape_ReceiveBuffer_From_SendBuffer(ReceBuffer,SendBuffer,h_Escape_SendCount, deviceCount , STEP );
Escape_Insert_ReceBuffer_Fellow_AtomInfoArray(d_atoms , ReceBuffer , h_Escape_ReceCount , h_LingerCount , h_ParticleCount ,STEP , deviceCount );


// Mapping Cell Table on GPUS
d_BuildShell_SendCount_Set_Zero(d_BuildShell_SendCount , deviceCount );
Load_Shell_Atoms_To_SendBuffer(d_atoms , h_ParticleCount ,SendBuffer ,deviceCount ,P ,T_Info ,Pnum ,d_BuildShell_SendCount ,STEP );


d_BuildShell_SendCount_DeviceToHost(d_BuildShell_SendCount, h_BuildShell_SendCount, deviceCount);
Build_Shell_ReceiveBuffer_From_SendBuffer(ReceBuffer , SendBuffer , h_BuildShell_SendCount , deviceCount , STEP );


Build_Shell_ReceiveCount_From_SendCount( d_BuildShell_ReceCount, d_BuildShell_SendCount , deviceCount);
Build_Shell_ReceCount_DeviceToHost( h_BuildShell_ReceCount ,  d_BuildShell_ReceCount , deviceCount) ;
Build_Shell_Insert_ReceBuffer_Fellow_AtomInfoArray(d_atoms , ReceBuffer , h_BuildShell_ReceCount , h_ParticleCount , h_AddShell_ParticleCount , STEP , deviceCount );

GlobalCellTable_Set_Zero( GlobalCellTable , T_Info , deviceCount );
Mapping_to_CellTable( d_atoms , GlobalCellTable , T_Info , h_AddShell_ParticleCount , deviceCount);


// Get force 
d_U_Set_Zero( d_U , deviceCount);
Computation_Force_Potential(d_atoms , h_ParticleCount ,GlobalCellTable , T_Info , d_U , U_lock , deviceCount);


// update kernel2
Update_AtomsInfo_kernel_2( d_atoms , h_ParticleCount , dt , deviceCount);


total_U = Get_Total_Potential(h_U, d_U, deviceCount);

 // Get the Kinetic on GPUs
 d_Ek_Set_Zero(d_EK, deviceCount);
 Computation_Kinetic(d_atoms , h_ParticleCount, d_EK, Signal, deviceCount);
 total_Ek =  Get_Total_Kinetic(h_EK, d_EK, deviceCount);
 printf("Step=%dth :\t kinetic is %f \t Potential is %f \n", step ,total_Ek/N , 0.5 * total_U / N );



 if(step<=Ne)
 {
   Scale_Velocity(d_atoms , h_ParticleCount , total_Ek * 2 , deviceCount , N);
 }


  All_devices_Synchronize(deviceCount);

 }



  free(atoms);
  free(Xseg);
  free(Yseg);
  free(Zseg);
  free(subdomain);

  for(int i=0;i<Pnum;i++)
    free(atomsInfoArr[i]);
  free(atomsInfoArr);

  free(h_ParticleCount);
  free(h_AddShell_ParticleCount);
  free(SubDomainBoundarySize);

  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_atoms[i]);
  }
  free(d_atoms);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(SendBuffer[i]);
  }
  free(SendBuffer);
  
  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(ReceBuffer[i]);
  } 
  free(ReceBuffer);

  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_BuildShell_SendCount[i]);
  }
  free(d_BuildShell_SendCount);


  for(int i=0;i<Pnum;i++)
    free(h_BuildShell_SendCount[i]);
  free(h_BuildShell_SendCount);

  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_BuildShell_ReceCount[i]);
  } 
  free(d_BuildShell_ReceCount);


  for(int i=0;i<Pnum;i++)
    free(h_BuildShell_ReceCount[i]);
  free(h_BuildShell_ReceCount);

  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(GlobalCellTable[i]);
  }
  free(GlobalCellTable);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_U[i]);
  }
  free(d_U);

  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(U_lock[i]);
  }
  free(U_lock);

  for(int i=0;i<Pnum;i++)
     free(h_U[i]);
  free(h_U);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_EK[i]);
  }
  free(d_EK);

  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(Signal[i]);
  }
  free(Signal);


  for(int i=0;i<Pnum;i++)
    free(h_EK[i]);
  free(h_EK);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_Adjust[i]);
  }
  free(d_Adjust);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_LingerCount[i]);
  }
  free(d_LingerCount);

  for(int i=0;i<Pnum;i++)
  {
    free(h_LingerCount[i]);
  }
  free( h_LingerCount);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_Escape_SendCount[i]);
  }
  free(d_Escape_SendCount);


  for(int i=0;i<Pnum;i++)
    free(h_Escape_SendCount[i]);

  free(h_Escape_SendCount);


  for(int i=0;i<Pnum;i++)
  {
    cudaSetDevice(i);
    cudaFree(d_Escape_ReceCount[i]);
  } 
  free(d_Escape_ReceCount);
  

  for(int i=0;i<Pnum;i++)
    free(h_Escape_ReceCount[i]);

  free(h_Escape_ReceCount);

}
