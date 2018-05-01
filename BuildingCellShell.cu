
void BocellNumSubdomain(int *SubdomainBoSize,subDomain *subdomain,int Pnum)
{
	 int outtemp = 0;
	 int intemp = 0;
	 for(int i=0;i<Pnum;i++)
	{
		outtemp = (subdomain[i].X.high - subdomain[i].X.low + 3)* (subdomain[i].Y.high - subdomain[i].Y.low + 3)*(subdomain[i].Z.high - subdomain[i].Z.low + 3);
		intemp = (subdomain[i].X.high - subdomain[i].X.low + 1)* (subdomain[i].Y.high - subdomain[i].Y.low + 1)*(subdomain[i].Z.high - subdomain[i].Z.low + 1);
		SubdomainBoSize[i] = outtemp - intemp;
	}
	
}





void d_BuildShell_SendCount_Set_Zero(int **d_BuildShell_SendCount,int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemset(d_BuildShell_SendCount[i], 0 , sizeof(int)*deviceCount ) ); //set SendCount[i] to 0
    } 
   // cudaMemset() 是同步调用函数
}


//  get the cell which subdomain it belong to and the sub domain id    ON GPU RUN
__device__ inline int d_cellToSubDomainId(int cx,int cy,int cz,TableInfo T_Info,deProcess P)
{
	int subDomain_id = -1;

    int Xcellnum = T_Info.cell_num[0];
	int Ycellnum = T_Info.cell_num[1];
	int Zcellnum = T_Info.cell_num[2];

	int XsubId = BLOCK_OWNER(cx,P.x,Xcellnum);
	int YsubId = BLOCK_OWNER(cy,P.y,Ycellnum);
	int ZsubId = BLOCK_OWNER(cz,P.z,Zcellnum);

    subDomain_id = XsubId + YsubId * P.x + ZsubId * P.x * P.y;
    return subDomain_id;
}




__global__ void Load_ShellAtoms_ToSendBuffer(AtomInfo *SubDomainAtomInfo,AtomInfo *SendBuffer,deProcess P,TableInfo T_Info,int subdomaimId,int Nsub,int Pnum,int *SendCount,int SegInterval)
{
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   int cx,cy,cz;
   int neibArr[PNUM];
   for(int i=0;i<Pnum;i++)
      neibArr[i] = 0;
      
   int xyzId = -1;
   int pbc_x;
   int pbc_y;
   int pbc_z;
   int i_offset ;
   
   if(id < Nsub)
   {
     //  get the atom cell coordinate
     cx = (int)(SubDomainAtomInfo[id].position[0] / T_Info.L_cell[0]);
     cy = (int)(SubDomainAtomInfo[id].position[1] / T_Info.L_cell[1]);
     cz = (int)(SubDomainAtomInfo[id].position[2] / T_Info.L_cell[2]);
    // Boundary Cell to Neibor Domain
	  for(int i=cx-1;i<=cx+1;i++)
	{
		  for(int j=cy-1;j<=cy+1;j++)
		{
		     for(int k=cz-1;k<=cz+1;k++)
		   {
              // pbc boundary 
		   	      pbc_x = (i + T_Info.cell_num[0]) % T_Info.cell_num[0];
              pbc_y = (j + T_Info.cell_num[1]) % T_Info.cell_num[1];
              pbc_z = (k + T_Info.cell_num[2]) % T_Info.cell_num[2]; 
              
			       xyzId = d_cellToSubDomainId(pbc_x ,pbc_y ,pbc_z ,T_Info ,P );
		   	  
		   	   if(xyzId != subdomaimId)
		   	  {
		   	  	neibArr[xyzId] = 1;   //tag which subDomain should send to
			  }
		   	    
		   }	
		}
	}
    
       for(int i=0;i<Pnum;i++)
      {
	       if(neibArr[i] != 0)
	     {
	        i_offset = atomicAdd(&SendCount[i],1); // get the index atomicly 

	        if( i_offset < SegInterval )
	        {
             SendBuffer[ SegInterval * i + i_offset ] = SubDomainAtomInfo[id];
	        }
	        else
	        {
	          printf("out of buffer\n");
	        }
	          
	     }
	 }
       
  } // endif 
  
}//no bug surely





void Load_Shell_Atoms_To_SendBuffer(AtomInfo ** d_atoms,int* h_ParticleCount,AtomInfo **SendBuffer,int deviceCount,deProcess P,TableInfo T_Info ,int Pnum,int **d_BuildShell_SendCount,int SegInterval)
{
     for(int i=0;i<deviceCount;i++) 
    { 
      int N = h_ParticleCount[i];   
      int dimgrid;
      dim3 dimBlock = 128;
      if (N % 128 == 0)
        dimgrid = N / 128;
      else
        dimgrid = N / 128 + 1;
	    dim3 dimGrid = dimgrid;
	  
      cudaSetDevice(i);
	    Load_ShellAtoms_ToSendBuffer<<<dimGrid,dimBlock>>>(d_atoms[i] ,SendBuffer[i] ,P ,T_Info ,i ,N ,Pnum ,d_BuildShell_SendCount[i] ,SegInterval );
      CUDA_CHECK_ERROR();
    } 

    All_devices_Synchronize(deviceCount);    // load data Synchronize！
  
}



void d_BuildShell_SendCount_DeviceToHost(int **d_BuildShell_SendCount,int ** h_BuildShell_SendCount, int deviceCount)
{
     for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemcpy(h_BuildShell_SendCount[i] , d_BuildShell_SendCount[i] ,sizeof(int) * deviceCount,cudaMemcpyDeviceToHost) );
    }
}






__global__ void test_loadShell_kernel(AtomInfo *d_atoms, int *SendCount,int deviceCount,int SegInterval, int thisdevice) // N : total !  count: check element number!
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;  // this atom's id
  if(id==0) // only one thread read data
  {
     printf("sendBuffer check the first 、the last and last next element:\n");
    
     int i;
     for(int j=0;j<deviceCount;j++)
     {
      printf("%d \n",SendCount[j]);
      if(j != thisdevice)
      {
        i = SegInterval * j +  SendCount[j];
        printf("%d:%f %f %f \t %d:%f %f %f \t %d:%f %f %f \n",SegInterval * j,
        d_atoms[0].position[0],d_atoms[0].position[1],d_atoms[0].position[2],
        i-1,d_atoms[i-1].position[0],d_atoms[i-1].position[1],d_atoms[i-1].position[2],
        i,d_atoms[i].position[0],d_atoms[i].position[1],d_atoms[i].position[2]);
        printf("\n\n");

      }

     }
    
  }  

}




void check_Load_Shell_Atoms_To_SendBuffer(AtomInfo **SendBuffer,int ** d_BuildShell_SendCount,int deviceCount,int SegInterval)
{

  for(int i=0;i<deviceCount;i++) 
  { 
    dim3 testBlock = 128;
    dim3 testGrid = 1;
    cudaSetDevice(i);
    test_loadShell_kernel<<<testGrid,testBlock>>>(SendBuffer[i], d_BuildShell_SendCount[i], deviceCount,SegInterval,i) ;
    CUDA_CHECK_ERROR();
  } 

}




// bug here adress in cuda 
void Build_Shell_ReceiveBuffer_From_SendBuffer(AtomInfo **ReceBuffer , AtomInfo **SendBuffer , int **h_BuildShell_SendCount , int deviceCount ,int SegInterval )
{
    for(int i=0;i<deviceCount;i++) 
  { 
      for(int j=0;j<deviceCount;j++)
     {
         if(j != i)
        {
          // CUDA_CALL( cudaMemcpyPeer( &(ReceBuffer[j][i*SegInterval]) , j , &(SendBuffer[i][j*SegInterval]) , i , sizeof(AtomInfo) * h_BuildShell_SendCount[i][j] )  );
           //CUDA_CALL( cudaMemcpyPeer( ReceBuffer[j] + i * SegInterval, j ,SendBuffer[i] + j * SegInterval , i , sizeof(AtomInfo) * h_BuildShell_SendCount[i][j] )  );
          CUDA_CALL( cudaMemcpyPeerAsync( &(ReceBuffer[j][i*SegInterval]) , j , &(SendBuffer[i][j*SegInterval]) , i , sizeof(AtomInfo) * h_BuildShell_SendCount[i][j] )  );
        } // 串行通信模拟
     }
  } 

}



void Build_Shell_ReceiveCount_From_SendCount(int ** d_BuildShell_ReceCount,int **d_BuildShell_SendCount , int deviceCount)
{
  for(int i=0;i<deviceCount;i++)  // ith gpu send
  { 
      for(int j=0;j<deviceCount;j++)
     {
        CUDA_CALL( cudaMemcpyPeer( &(d_BuildShell_ReceCount[i][j]) , i , &(d_BuildShell_SendCount[j][i]) , j , sizeof(int) )  );
     } // 串行通信
  } 

}



void Build_Shell_ReceCount_DeviceToHost(int ** h_BuildShell_ReceCount , int ** d_BuildShell_ReceCount , int deviceCount)
{
    for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL( cudaMemcpy( h_BuildShell_ReceCount[i] , d_BuildShell_ReceCount[i] , sizeof(int) * deviceCount , cudaMemcpyDeviceToHost ) );
    }
}




void Build_Shell_Insert_ReceBuffer_Fellow_AtomInfoArray(AtomInfo **d_atoms , AtomInfo **ReceBuffer , int ** h_BuildShell_ReceCount , int* h_ParticleCount ,int *h_AddShell_ParticleCount , int SegInterval ,int deviceCount )
{
    for(int i=0;i<deviceCount;i++)
    {
      int offset_atomArry = 0;
       for(int j=0;j<deviceCount;j++)
      {
        if(j != i)
        {
          cudaSetDevice(i);
          CUDA_CALL( cudaMemcpyAsync( &(d_atoms[i][h_ParticleCount[i] + offset_atomArry]) , &(ReceBuffer[i][SegInterval * j]) , sizeof(AtomInfo) * h_BuildShell_ReceCount[i][j] , cudaMemcpyDeviceToDevice)  );
          //CUDA_CALL( cudaMemcpy( &(d_atoms[i][h_ParticleCount[i] + offset_atomArry]) , &(ReceBuffer[i][SegInterval * j]) , sizeof(AtomInfo) * h_BuildShell_ReceCount[i][j] , cudaMemcpyDeviceToDevice)  );
          offset_atomArry = offset_atomArry + h_BuildShell_ReceCount[i][j];
        }

      }

      h_AddShell_ParticleCount[i] = h_ParticleCount[i] + offset_atomArry;
      
    } //串行通信模式

} 





__global__ void test_Insert_kernel(AtomInfo *d_atoms, AtomInfo *ReceBuffer ,int * d_ReceCount ,int dev_id , int deviceCount,int SegInterval, int addShellCount )
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


void Check_Insert_ReceBuffer_Fellow_AtomInfoArray(AtomInfo **d_atoms , AtomInfo **ReceBuffer , int ** d_BuildShell_ReceCount  ,int *h_AddShell_ParticleCount , int SegInterval ,int deviceCount )
{
  for(int i=0;i<deviceCount;i++)
  {
    dim3 testBlock = 128;
    dim3 testGrid = 1;
    cudaSetDevice(i);
    test_Insert_kernel<<<testGrid,testBlock>>>(d_atoms[i], ReceBuffer[i] , d_BuildShell_ReceCount[i] , i ,  deviceCount, SegInterval, h_AddShell_ParticleCount[i] );
    CUDA_CHECK_ERROR(); 
  }

}





