void MapParticleToSubdomain(AtomInfo *atoms,AtomInfo **atomsInfoArr,int MallocLen,TableInfo T_Info,deProcess P,int* h_ParticleCount,int n,int Pnum)
{
	double x,y,z;
	int cx,cy,cz;
	int subDomainId = -1;
	
	  for(int i=0; i<n;i++)  // n is the total atom number in the system-
	{
	    x = atoms[i].position[0];
	  	y = atoms[i].position[1];
	  	z = atoms[i].position[2];
	  	
		cx = (int)(x / T_Info.L_cell[0]);
		cy = (int)(y / T_Info.L_cell[1]);
		cz = (int)(z / T_Info.L_cell[2]); 
		
		subDomainId = GetCellSubDomainId(cx,cy,cz,T_Info,P);
		
        atomsInfoArr[subDomainId][h_ParticleCount[subDomainId]] = atoms[i]; // insert the atom to the right subdomain 
    
        h_ParticleCount[subDomainId]++;   // the counter add 1
    
		if( h_ParticleCount[subDomainId] >= MallocLen ) // out of memory check !!!
		{
			printf("the subdomain atom out of the buffer!\n");
			exit(1);
		}
		
	}
	

}



void Check_MapParticleToSubdomain(AtomInfo **atomsInfoArr,TableInfo T_Info,deProcess P,int* h_ParticleCount,int Pnum)
{
   int Particle_sum = 0;

   for(int i=0;i<Pnum;i++ )
  {
      Particle_sum = Particle_sum + h_ParticleCount[i];
      int count = 0;
      for(int j=0;j<h_ParticleCount[i];j++)
      {
         double x = atomsInfoArr[i][j].position[0];
         double y = atomsInfoArr[i][j].position[1];
         double z = atomsInfoArr[i][j].position[2];

         int cx = (int)(x / T_Info.L_cell[0]);
		 int cy = (int)(y / T_Info.L_cell[1]);
		 int cz = (int)(z / T_Info.L_cell[2]); 
         
         int subDomainId = GetCellSubDomainId(cx,cy,cz,T_Info,P);
         if(subDomainId == i)
         {
            count++;
         }  else{
              printf("the %dth partition error\n ",i);
         }


      }

      if(count ==h_ParticleCount[i])
        printf("%dth Partition OK!\n",i);
      printf("check the first the last and the last next element:\n");
      printf("%f %f %f \t %f %f %f \t %f %f %f \n",atomsInfoArr[i][0].position[0], atomsInfoArr[i][0].position[1], atomsInfoArr[i][0].position[2]
      ,atomsInfoArr[i][h_ParticleCount[i]-1].position[0], atomsInfoArr[i][h_ParticleCount[i]-1].position[1], atomsInfoArr[i][h_ParticleCount[i]-1].position[2] ,
      atomsInfoArr[i][h_ParticleCount[i]].position[0], atomsInfoArr[i][h_ParticleCount[i]].position[1], atomsInfoArr[i][h_ParticleCount[i]].position[2]);  
          
  }

  printf("the total Particel is %d\n",Particle_sum);

}





// -------------memeroy testing kernel----------------
__global__ void test_kernel(AtomInfo *d_atoms, int N) // N : total !  count: check element number!
{
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;  // this atom's id
  if(id==0) // only one thread read data
  {
     printf("check the first 、the last and last next element:\n");
     int  i = N;
     printf("%f %f %f \t %f %f %f \t %f %f %f \n",
         d_atoms[0].position[0],d_atoms[0].position[1],d_atoms[0].position[2],
         d_atoms[i-1].position[0],d_atoms[i-1].position[1],d_atoms[i-1].position[2],
		     d_atoms[i].position[0],d_atoms[i].position[1],d_atoms[i].position[2]);

    printf("\n\n");

  }  

}


void test_Copy(AtomInfo **d_atoms,int *h_ParticleCount,int deviceCount)
{
  //=========== check the force computation =======
  for(int i=0;i<deviceCount;i++)
 { 
    dim3 testBlock = 128;
    dim3 testGrid = 1;
    int N =  h_ParticleCount[i];
  
    cudaSetDevice(i);
    test_kernel<<<testGrid,testBlock>>>(d_atoms[i],N);
    CUDA_CHECK_ERROR();
 }

}

