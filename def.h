
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>

// l_x = 17
#define Dim 3
#define FCC_NUM 4
#define L_x 100
#define L_y 100
#define L_z 100
#define a_x 5.45
#define a_y 5.45
#define a_z 5.45
#define NE 50
#define NP 50
#define NS 20
#define RCUT (10.0)
#define Rc 10.0

#define DT 5.0
#define Termp 80.0
#define K_B 8.625e-5
#define Mass 40.0
#define Max_Mem 512
#define MaxSize 256

#define  rh  64       //particle density of cell
#define  SCALER 2 
#define  PNUM 2     //s processor number 


#define EPSILON 1.032e-2
#define SIGMA 3.405
#define TIME_UNIT_CONVERSION 10.18

#define sigma_6 1558.485254168438765
#define sigma_12 2428876.287460463180885420

// how to split the data for cell on x,y,z axis
#define BLOCK_LOW(id,p,n) ( (id)*(n)/(p) )
#define BLOCK_HIGH(id,p,n) ( BLOCK_LOW((id)+1,p,n) - 1 )
#define BLOCK_SIZE(id,p,n)  ( BLOCK_LOW((id)+1,p,n) - BLOCK_LOW(id,p,n) )
#define BLOCK_OWNER(index,p,n) ( ( (p) * ((index)+1) - 1 )/(n) )



#define CUDA_CALL(x)                                                           \
  {                                                                            \
    const cudaError_t e = (x);                                                 \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "\n Cuda Error: %s (error num=%d at %s:%d)\n",           \
              cudaGetErrorString(e), e, __FILE__, __LINE__);                   \
      cudaDeviceReset();                                                       \
      assert(0);                                                               \
    }                                                                          \
  }           





#define CUDA_CHECK_ERROR()                                                     \
  do {                                                                         \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "\n Cuda Error: %s (error num=%d at %s:%d)\n",           \
              cudaGetErrorString(e), e, __FILE__, __LINE__);                   \
      cudaDeviceReset();                                                       \
      assert(0);                                                               \
    }                                                                          \
  } while (0)
  
