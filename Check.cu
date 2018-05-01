
void check_copy( AtomInfo *check_atoms , AtomInfo **d_atoms , int* h_ParticleCount , int deviceCount )
{
    int offset = 0;
    for(int i=0;i<deviceCount;i++)
    {
      cudaSetDevice(i);
      CUDA_CALL(cudaMemcpy( &check_atoms[offset] , d_atoms[i] , sizeof(AtomInfo) * h_ParticleCount[i] , cudaMemcpyDeviceToHost ) );
      offset = offset + h_ParticleCount[i];
    } 

}


void Sort_atoms(AtomInfo *check_atoms , int N,int step, int deviceCount)
{
    AtomInfo *Sort_atoms = (AtomInfo *)malloc(sizeof(AtomInfo) * N); 
    
    for(int i=0;i<N;i++)
    {
        Sort_atoms[check_atoms[i].atom_id] = check_atoms[i] ; 
    }
    
    char file[1024];
    sprintf(file,"%d_%d_step_check.data",deviceCount,step);
    FILE *fp = fopen(file, "wb");


    for(int i=0;i<N;i++)
    {
        // V23// fprintf(fp,"%d\t%lf\t%lf\t%lf\t\n", i , Sort_atoms[i].position[0], Sort_atoms[i].position[1] , Sort_atoms[i].position[2]);
        // V24// fprintf(fp,"%d\t%lf\t%lf\t%lf\t\n", i , Sort_atoms[i].velocity[0], Sort_atoms[i].velocity[1] , Sort_atoms[i].velocity[2]);
       // fprintf(fp,"%d\t%lf\t%lf\t%lf\t\n", i , Sort_atoms[i].force[0], Sort_atoms[i].force[1] , Sort_atoms[i].force[2]);
        fprintf(fp,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t\n", i , Sort_atoms[i].position[0], Sort_atoms[i].position[1] , Sort_atoms[i].position[2],
        Sort_atoms[i].velocity[0], Sort_atoms[i].velocity[1] , Sort_atoms[i].velocity[2]);
        
    }
      


    free(Sort_atoms);
    fclose(fp);

}





