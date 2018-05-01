void d_LingerCount_Set_Zero(int **d_LingerCount,int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
  {
    cudaSetDevice(i);
    CUDA_CALL( cudaMemset(d_LingerCount[i] , 0 , sizeof(int) ) ); //set ReceCount[i] to 0
  }

}


void d_Escape_SendCount_Set_Zero(int **d_Escape_SendCount , int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemset(d_Escape_SendCount[i] , 0 , sizeof(int)*deviceCount ) ); //set SendCount[i] to 0
    } 
   // cudaMemset() 是同步调用函数
}





__global__ void Adjust_kernel(AtomInfo *d_atoms , int N , AtomInfo *d_Adjust ,int *d_LingerCount, AtomInfo *SendBuffer ,int *d_Escape_SendCount , int SegInterval , deProcess P , TableInfo T_Info, int subDomainId,int Pnum )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int cx,cy,cz;
  int J_subDomainId;
  int atomInf_offset;
  int send_offset;

  if(id < N)
  {
      cx = (int)(d_atoms[id].position[0] / T_Info.L_cell[0]);  //  get the atom cell coordinate
      cy = (int)(d_atoms[id].position[1] / T_Info.L_cell[1]);
      cz = (int)(d_atoms[id].position[2] / T_Info.L_cell[2]);

      J_subDomainId = d_cellToSubDomainId(cx , cy , cz , T_Info , P );
    
      if(J_subDomainId == subDomainId)              // not escape atom
      {
        atomInf_offset = atomicAdd(&(d_LingerCount[0]) , 1); // get the adjustment array index parallel
        d_Adjust[atomInf_offset] = d_atoms[id];      // insert into  d_Adjust array when it is not escape 
      }
      else
      {
           send_offset = atomicAdd(&d_Escape_SendCount[J_subDomainId],1);
           if( send_offset < SegInterval )
	      {
              SendBuffer[ SegInterval * J_subDomainId + send_offset ] = d_atoms[id];
	      }
	       else
	      {
	           printf("out of buffer\n");
	      }
      }
  }

}



// int* h_ParticleCount

void Split_Escape_And_linger(AtomInfo ** d_atoms ,int* h_ParticleCount , AtomInfo **d_Adjust, int **d_LingerCount ,AtomInfo **SendBuffer, int **d_Escape_SendCount ,int deviceCount,deProcess P,TableInfo T_Info ,int Pnum ,int SegInterval)
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
    Adjust_kernel<<<dimGrid,dimBlock>>>(d_atoms[i] , N , d_Adjust[i] , d_LingerCount[i], SendBuffer[i] , d_Escape_SendCount[i] , SegInterval , P , T_Info, i, Pnum );

    CUDA_CHECK_ERROR();

   } //send the escape atom to sendbuffer and the linger atom to the d_Adjust array
 
   All_devices_Synchronize(deviceCount);

}



void d_Escape_SendCount_DeviceToHost(int **d_Escape_SendCount ,int ** h_Escape_SendCount , int deviceCount)
{
     for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemcpy(h_Escape_SendCount[i] , d_Escape_SendCount[i] , sizeof(int) * deviceCount , cudaMemcpyDeviceToHost) );
    }

}



void test_d_Escape_SendCount(int ** h_Escape_SendCount , int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
    {
        printf("the %dth GPU escape particle number is : ",i);
         for(int j=0;j<deviceCount;j++)
        {
            printf("%d\t",h_Escape_SendCount[i][j]);
        }
        printf("\n");
    }

}



void d_Escape_ReceiveCount_From_SendCount(int ** d_Escape_ReceCount,int **d_Escape_SendCount , int deviceCount)
{
  for(int i=0;i<deviceCount;i++)  // ith gpu send
  { 
      for(int j=0;j<deviceCount;j++)
     {
        CUDA_CALL( cudaMemcpyPeer( &(d_Escape_ReceCount[i][j]) , i , &(d_Escape_SendCount[j][i]) , j , sizeof(int) )  );
     } // 串行通信
  } 

}



void d_Escape_ReceCount_DeviceToHost(int ** h_Escape_ReceCount , int ** d_Escape_ReceCount , int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemcpy( h_Escape_ReceCount[i] , d_Escape_ReceCount[i] , sizeof(int) * deviceCount , cudaMemcpyDeviceToHost ) );
    }
    
}



void LingerCount_DeviceToHost(int **h_LingerCount , int **d_LingerCount,int deviceCount)
{
  for(int i=0;i<deviceCount;i++)
  {
    cudaSetDevice(i);
    CUDA_CALL( cudaMemcpy( h_LingerCount[i] , d_LingerCount[i] , sizeof(int) , cudaMemcpyDeviceToHost ) );
  }

}




void linger_To_AtomInfoArray(AtomInfo ** d_atoms , AtomInfo **d_Adjust , int **h_LingerCount ,int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
   {
     cudaSetDevice(i);
     CUDA_CALL( cudaMemcpy( d_atoms[i] , d_Adjust[i] , sizeof(AtomInfo) * h_LingerCount[i][0] , cudaMemcpyDeviceToDevice)  );
   }

}





__global__ void test_linger_To_AtomInfo_kernel(AtomInfo *d_atoms, AtomInfo *d_Adjust ,int *d_LingerCount ,int thisDevice)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;  //this atom's id

  if( id == 0 )
  {
      int count = 0 ;  
      for(int i=0;i<d_LingerCount[0];i++)
    {
        bool b1 = ( d_atoms[i].velocity[0] == d_Adjust[i].velocity[0] );
        bool b2 = ( d_atoms[i].velocity[1] == d_Adjust[i].velocity[1] );
        bool b3 = ( d_atoms[i].velocity[2] == d_Adjust[i].velocity[2] );

        if(b1&&(b2&&b3))
           count ++;
      
    }
     if(count==d_LingerCount[0])
         printf("\n\nthe %dth gpu check linger insert %d number partcle is successfully! \n\n ",thisDevice,count);
      else
        printf("\nlinger error on %dth GPU\n",thisDevice);

  }


}



void test_linger_To_AtomInfo(AtomInfo **d_atoms, AtomInfo **d_Adjust ,int **d_LingerCount , int deviceCount)
{
  for(int i=0;i<deviceCount;i++)
  {
    dim3 testBlock = 128;
    dim3 testGrid = 1;
    cudaSetDevice(i);
    test_linger_To_AtomInfo_kernel<<<testGrid,testBlock>>>(d_atoms[i], d_Adjust[i] ,d_LingerCount[i] , i);
    CUDA_CHECK_ERROR(); 
  }

}





// bug here adress in cuda 
void Escape_ReceiveBuffer_From_SendBuffer(AtomInfo **ReceBuffer , AtomInfo **SendBuffer , int **h_Escape_SendCount , int deviceCount ,int SegInterval )
{
    for(int i=0;i<deviceCount;i++) 
  { 
      for(int j=0;j<deviceCount;j++)
     {
         if(j != i)
        {
          
          CUDA_CALL( cudaMemcpyPeerAsync( &(ReceBuffer[j][i*SegInterval]) , j , &(SendBuffer[i][j*SegInterval]) , i , sizeof(AtomInfo) * h_Escape_SendCount[i][j] )  );
          //CUDA_CALL( cudaMemcpyPeer( &(ReceBuffer[j][i*SegInterval]) , j , &(SendBuffer[i][j*SegInterval]) , i , sizeof(AtomInfo) * h_Escape_SendCount[i][j] )  );
        } // 串行通信模拟
     }
  } 

}


// int ** h_Escape_ReceCount 
void Escape_Insert_ReceBuffer_Fellow_AtomInfoArray(AtomInfo **d_atoms , AtomInfo **ReceBuffer , int ** h_Escape_ReceCount , int **h_LingerCount , int* h_ParticleCount , int SegInterval ,int deviceCount )
{
    for(int i=0;i<deviceCount;i++)
    {
      int offset_atomArry = 0;
       for(int j=0;j<deviceCount;j++)
      {
        if(j != i)
        {
          cudaSetDevice(i);
          CUDA_CALL( cudaMemcpyAsync( &(d_atoms[i][h_LingerCount[i][0] + offset_atomArry]) , &(ReceBuffer[i][SegInterval * j]) , sizeof(AtomInfo) * h_Escape_ReceCount[i][j] , cudaMemcpyDeviceToDevice)  );
          //CUDA_CALL( cudaMemcpy( &(d_atoms[i][h_LingerCount[i][0] + offset_atomArry]) , &(ReceBuffer[i][SegInterval * j]) , sizeof(AtomInfo) * h_Escape_ReceCount[i][j] , cudaMemcpyDeviceToDevice)  );
          offset_atomArry = offset_atomArry + h_Escape_ReceCount[i][j];
        }

      }

      h_ParticleCount[i] = h_LingerCount[i][0] + offset_atomArry;
      
    } //串行通信模式

} 







__global__ void test_Escape_Insert_kernel(AtomInfo *d_atoms, AtomInfo *ReceBuffer ,int * d_ReceCount ,int dev_id , int deviceCount,int SegInterval, int addShellCount )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;  // this atom's id
  if(id==0) // only one thread read data
  {
    if(dev_id != (deviceCount-1))
    {

      if( d_atoms[addShellCount-1].position[0] == ReceBuffer[ SegInterval*(deviceCount-1) + d_ReceCount[deviceCount-1] - 1].position[0])
         printf("the %dth Insert sucessfully !\n",dev_id); 
      else
         printf("the %dth Insert error !\n",dev_id);
    }
    else
    {

      if(d_atoms[addShellCount - 1].position[0] == ReceBuffer[ SegInterval*(deviceCount-1-1) + d_ReceCount[deviceCount-1-1]-1].position[0])
         printf("the %dth Insert sucessfully !\n",dev_id); 
      else
         printf("the %dth Insert error !\n",dev_id);

    }

    
  }  


}



void Check_Escape_Insert_ReceBuffer_Fellow_AtomInfoArray(AtomInfo **d_atoms , AtomInfo **ReceBuffer , int ** d_BuildShell_ReceCount  ,int *h_AddShell_ParticleCount , int SegInterval ,int deviceCount )
{
  for(int i=0;i<deviceCount;i++)
  {
    dim3 testBlock = 128;
    dim3 testGrid = 1;
    cudaSetDevice(i);
    test_Escape_Insert_kernel<<<testGrid,testBlock>>>(d_atoms[i], ReceBuffer[i] , d_BuildShell_ReceCount[i] , i ,  deviceCount, SegInterval, h_AddShell_ParticleCount[i] );
    CUDA_CHECK_ERROR(); 
  }

}




