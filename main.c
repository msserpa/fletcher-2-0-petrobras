#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "boundary.h"
#include "source.h"
#include "utils.h"
#include "map.h"
#include "driver.h"
#include "fletcher.h"
#include "model.h"

#ifdef UNIFIED
  #include "CUDA/cuda_stuff.h"
  #include <cuda_runtime.h>
  #include <nvToolsExt.h>
  int deviceCount;
#endif

enum Form {ISO, VTI, TTI};


int main(int argc, char** argv){
  enum Form prob;        // problem formulation
  int nx;                // grid points in x
  int ny;                // grid points in y
  int nz;                // grid points in z
  int bord = 4;          // border size to apply the stencil at grid extremes
  int absorb;            // absortion zone size
  int sx;                // grid dimension in x (grid points + 2*border + 2*absortion)
  int sy;                // grid dimension in y (grid points + 2*border + 2*absortion)
  int sz;                // grid dimension in z (grid points + 2*border + 2*absortion)
  int st;                // number of time steps
  float dx;              // grid step in x
  float dy;              // grid step in y
  float dz;              // grid step in z
  float dt;              // time advance at each time step
  float tmax;            // desired simulation final time
  int ixSource;          // source x index
  int iySource;          // source y index
  int izSource;          // source z index
  int iSource;           // source index (ix,iy,iz) maped into 1D array
  //PPL  int i, ix, iy, iz, it; // for indices
  int i, it;             // for indices
  //PPL  char fNameAbs[128];    // prefix of absortion file
  char fNameSec[128];    // prefix of sections files

  const int rank = 0;
  const float dtOutput = 0.01;

  it = 0; //PPL

  // input problem definition

  if(argc < ARGS){
    printf("program requires %d input arguments; execution halted\n", ARGS - 1);
    exit(-1);
  } 
  strcpy(fNameSec, argv[1]);
  nx = atoi(argv[2]);
  ny = atoi(argv[3]);
  nz = atoi(argv[4]);
  absorb = atoi(argv[5]);
  dx = atof(argv[6]);
  dy = atof(argv[7]);
  dz = atof(argv[8]);
  dt = atof(argv[9]);
  tmax = atof(argv[10]);

  // verify problem formulation

  if(strcmp(fNameSec,"ISO") == 0)
    prob = ISO;
  else if(strcmp(fNameSec,"VTI") == 0)
    prob = VTI;
  else if(strcmp(fNameSec,"TTI") == 0)
    prob = TTI;
  else{
    printf("Input problem formulation (%s) is unknown\n", fNameSec);
    exit(-1);
  }

  #ifdef _DUMP
    printf("Problem is ");
    switch (prob){
      case ISO:
        printf("isotropic\n");
        break;
      case VTI:
        printf("anisotropic with vertical transversely isotropy using sigma=%f\n", SIGMA);
        break;
      case TTI:
        printf("anisotropic with tilted transversely isotropy using sigma=%f\n", SIGMA);
        break;
    }
  #endif

  // grid dimensions from problem size

  sx = nx + 2 * bord + 2 * absorb;
  sy = ny + 2 * bord + 2 * absorb;
  sz = nz + 2 * bord + 2 * absorb;

  // number of time iterations

  st = ceil(tmax / dt);

  // source position

  ixSource = sx / 2;
  iySource = sy / 2;
  izSource = sz / 2;
  iSource = ind(ixSource, iySource, izSource);

  // dump problem input data

  #ifdef _DUMP
    printf("Grid size is (%d,%d,%d) with spacing (%.2f,%.2f,%.2f); simulated area (%.2f,%.2f,%.2f) \n", 
      nx, ny, nz, dx, dy, dz, (nx - 1) * dx, (ny - 1) * dy, (nz - 1) * dz);
    printf("Grid is extended by %d absortion points and %d border points at each extreme\n", absorb, bord);
    printf("Wave is propagated at internal+absortion points of size (%d,%d,%d)\n",
      nx + 2 * absorb, ny + 2 * absorb, nz + 2 * absorb);
    printf("Source at coordinates (%d,%d,%d)\n", ixSource, iySource, izSource);
    printf("Will run %d time steps of %f to reach time %f\n", st, dt, st * dt);

    #ifdef _OPENMP
      #pragma omp parallel
      #pragma omp single
        printf("Execution with %d OpenMP threads\n", omp_get_num_threads());
    #else
      printf("Sequential execution\n");
    #endif
  #endif

  nvtxRangePushA("GPUInitialize");
  #ifdef UNIFIED
    DRIVER_GPUInitialize();
  #endif
  nvtxRangePop();


  nvtxRangePushA("MemoryAlloc");
  // allocate input anisotropy arrays

  float *vpz = NULL;      // p wave speed normal to the simetry plane
  #ifdef UNIFIED
    cudaMallocManaged((void **) &vpz, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    vpz = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *vsv = NULL;      // sv wave speed normal to the simetry plane
  #ifdef UNIFIED
    cudaMallocManaged((void **) &vsv, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    vsv = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *epsilon = NULL;  // Thomsen isotropic parameter
  #ifdef UNIFIED
    cudaMallocManaged((void **) &epsilon, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    epsilon = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *delta = NULL;    // Thomsen isotropic parameter
  #ifdef UNIFIED
    cudaMallocManaged((void **) &delta, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    delta = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *phi = NULL;     // isotropy simetry azimuth angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &phi, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    phi = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *theta = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &theta, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    theta = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  // input anisotropy arrays for selected problem formulation

    switch(prob){

    case ISO:

      for(i = 0; i < sx * sy * sz; i++){
        vpz[i] = 3000.0;
        epsilon[i] = 0.0;
        delta[i] = 0.0;
        phi[i] = 0.0;
        theta[i] = 0.0;
        vsv[i] = 0.0;
      }
      break;

    case VTI:

      if(SIGMA > MAX_SIGMA){
        printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n", 
          SIGMA, MAX_SIGMA);
      }
      for(i = 0; i < sx * sy * sz; i++){
        vpz[i] = 3000.0;
        epsilon[i] = 0.24;
        delta[i] = 0.1;
        phi[i] = 0.0;
        theta[i] = 0.0;
        if(SIGMA > MAX_SIGMA)
          vsv[i] = 0.0;
        else
          vsv[i] = vpz[i] * sqrtf(fabsf(epsilon[i] - delta[i]) / SIGMA);
      }
      break;

    case TTI:

      if(SIGMA > MAX_SIGMA){
        printf("Since sigma (%f) is greater that threshold (%f), sigma is considered infinity and vsv is set to zero\n", 
          SIGMA, MAX_SIGMA);
      }
      #ifndef UNIFIED
        for(i = 0; i < sx * sy * sz; i++){
          vpz[i] = 3000.0;
          epsilon[i] = 0.24;
          delta[i] = 0.1;
          phi[i] = 0.0;
          theta[i] = atanf(1.0);
          if(SIGMA > MAX_SIGMA)
            vsv[i] = 0.0;
          else
            vsv[i] = vpz[i] * sqrtf(fabsf(epsilon[i] - delta[i]) / SIGMA);
        }
      #endif
  } // end switch

  // coeficients of derivatives at H1 operator

  float *ch1dxx = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &ch1dxx, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    ch1dxx = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *ch1dyy = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &ch1dyy, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    ch1dyy = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *ch1dzz = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &ch1dzz, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    ch1dzz = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *ch1dxy = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &ch1dxy, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    ch1dxy = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *ch1dyz = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &ch1dyz, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    ch1dyz = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *ch1dxz = NULL;  // isotropy simetry deep angle
  #ifdef UNIFIED
    cudaMallocManaged((void **) &ch1dxz, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    ch1dxz = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  #ifndef UNIFIED
    float sinTheta, cosTheta, sin2Theta, sinPhi, cosPhi, sin2Phi;
    for(i = 0; i < sx * sy * sz; i++){
      sinTheta = sin(theta[i]);
      cosTheta = cos(theta[i]);
      sin2Theta = sin(2.0 * theta[i]);
      sinPhi = sin(phi[i]);
      cosPhi = cos(phi[i]);
      sin2Phi = sin(2.0 * phi[i]);
      ch1dxx[i] = sinTheta * sinTheta * cosPhi *cosPhi;
      ch1dyy[i] = sinTheta * sinTheta * sinPhi *sinPhi;
      ch1dzz[i] = cosTheta * cosTheta;
      ch1dxy[i] = sinTheta * sinTheta * sin2Phi;
      ch1dyz[i] = sin2Theta           * sinPhi;
      ch1dxz[i] = sin2Theta           * cosPhi;
    }
    #ifdef _DUMP
      printf("ch1dxx[0] = %f; ch1dyy[0] = %f; ch1dzz[0] = %f; ch1dxy[0] = %f; ch1dxz[0] = %f; ch1dyz[0] = %f\n",
        ch1dxx[0], ch1dyy[0], ch1dzz[0], ch1dxy[0], ch1dxz[0], ch1dyz[0]);
    #endif
  #endif

  // pressure fields at previous, current and future time steps

  float *pp = NULL;
  #ifdef UNIFIED
    cudaMallocManaged((void **) &pp, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    pp = (float *) malloc(sx * sy * sz * sizeof(float)); 
  #endif

  float *pc = NULL;
  #ifdef UNIFIED
    cudaMallocManaged((void **) &pc, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    pc = (float *) malloc(sx * sy * sz * sizeof(float)); 
  #endif

  float *qp = NULL;
  #ifdef UNIFIED
    cudaMallocManaged((void **) &qp, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    qp = (float *) malloc(sx * sy * sz * sizeof(float)); 
  #endif

  float *qc = NULL;
  #ifdef UNIFIED
    cudaMallocManaged((void **) &qc, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    qc = (float *) malloc(sx * sy * sz * sizeof(float)); 
  #endif

  #ifndef UNIFIED
    for(i = 0; i < sx * sy * sz; i++){
      pp[i] = 0.0f; pc[i] = 0.0f;
      qp[i] = 0.0f; qc[i] = 0.0f;
    }
  #endif

  float *fatAbsorb = NULL;
  // absortion zone
  #ifdef _ABSOR_SQUARE
    #ifdef UNIFIED
      cudaMallocManaged((void **) &fatAbsorb, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
    #else
      fatAbsorb = (float *) malloc(sx * sy * sz * sizeof(float));
    #endif  
    CreateSquareAbsorb(sx, sy, sz, nx, ny, nz, bord, absorb, dx, dy, dz, fatAbsorb);
  #endif

  #ifdef _ABSOR_SPHERE
    #ifdef UNIFIED
      cudaMallocManaged((void **) &fatAbsorb, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
    #else
      fatAbsorb = (float *) malloc(sx * sy * sz * sizeof(float));
    #endif  
    CreateSphereAbsorb(sx, sy, sz, nx, ny, nz, bord, absorb, dx, dy, dz, fatAbsorb);
  #endif

  #ifndef UNIFIED
  #ifdef _DUMP
    if(fatAbsorb != NULL)
    DumpFieldToFile(sx, sy, sz, 0, sx - 1, 0, sy - 1, 0, sz - 1, dx, dy, dz, fatAbsorb, "Absorb");
  #endif

  /* NEW KERNEL!!! */

  // stability condition

    float maxvel;
    maxvel = vpz[0] * sqrt(1.0 + 2 * epsilon[0]);
    for(i = 1; i < sx * sy * sz; i++)
      maxvel = fmaxf(maxvel, vpz[i] * sqrt(1.0 + 2 * epsilon[i]));
    
    float mindelta = dx;
    if(dy < mindelta)
      mindelta = dy;
    if(dz < mindelta)
      mindelta = dz;
    float recdt;
    recdt = (MI * mindelta) / maxvel;
    
    #ifdef _DUMP
      printf("Recomended maximum time step is %f; used time step is %f\n", recdt, dt);
    #endif
  
    // random boundary speed
    #ifdef _RANDOM_BDRY
      RandomVelocityBoundary(sx, sy, sz, nx, ny, nz, bord, absorb, vpz, vsv);
    #endif

    #ifdef _DUMP
      DumpFieldToFile(sx, sy, sz, 0, sx - 1, 0, sy - 1, 0, sz - 1, dx, dy, dz, vpz, "Velocity");
    #endif

  #endif
  // coeficients of H1 and H2 at PDEs

  float *v2px = NULL;  // coeficient of H2(p)
  #ifdef UNIFIED
    cudaMallocManaged((void **) &v2px, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    v2px = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *v2pz = NULL;  // coeficient of H1(q)
  #ifdef UNIFIED
    cudaMallocManaged((void **) &v2pz, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else  
    v2pz = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *v2sz = NULL;  // coeficient of H1(p-q) and H2(p-q)
  #ifdef UNIFIED
    cudaMallocManaged((void **) &v2sz, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    v2sz = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

  float *v2pn = NULL;  // coeficient of H2(p)
  #ifdef UNIFIED
    cudaMallocManaged((void **) &v2pn, sx * sy * sz * sizeof(float), cudaMemAttachGlobal);
  #else
    v2pn = (float *) malloc(sx * sy * sz * sizeof(float));
  #endif

#ifndef UNIFIED
  for(i = 0; i < sx * sy * sz; i++){
    v2sz[i] = vsv[i] * vsv[i];
    v2pz[i] = vpz[i] * vpz[i];
    v2px[i] = v2pz[i] * (1.0 + 2.0 * epsilon[i]);
    v2pn[i] = v2pz[i] * (1.0 + 2.0 * delta[i]);
  }

  #ifdef _DUMP
    printf("v2sz[0] = %f; v2pz[0] = %f; v2px[0] = %f; v2pn[0] = %f\n", v2sz[0], v2pz[0], v2px[0], v2pn[0]);
  #endif
#endif
  // slices

  nvtxRangePop();

  nvtxRangePushA("ArraysInit");
  #ifdef UNIFIED
    DRIVER_ArraysInit(vpz, vsv, epsilon, delta, phi, theta, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, pp, pc, qp, qc,
      NULL, v2px, v2pz, v2sz, v2pn, sx * sy * sz);
  #endif
  nvtxRangePop();

  nvtxRangePushA("OpenSlice");
  //PPL  char fName[10];
  int ixStart = 0;
  int ixEnd = sx - 1;
  int iyStart = 0;
  int iyEnd = sy - 1;
  int izStart = 0;
  int izEnd = sz - 1;

  SlicePtr sPtr;
  sPtr = OpenSliceFile(ixStart, ixEnd, iyStart, iyEnd, izStart, izEnd, dx, dy, dz, dt, fNameSec);

  DumpSliceFile(sx, sy, sz, pc, sPtr);
  #ifdef _DUMP
    DumpSlicePtr(sPtr);
    DumpSliceSummary(sx, sy, sz, sPtr, dt, it, pc, 0);
  #endif
  nvtxRangePop();

  // Model do:
  // - Initialize
  // - time loop
  // - calls Propagate
  // - calls TimeForward
  // - calls InsertSource
  // - do AbsorbingBoundary and DumpSliceFile, if needed
  // - Finalize
  nvtxRangePushA("Model");
  Model(st, iSource, dtOutput, sPtr, sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz,
    v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, vpz, vsv, epsilon, delta, phi, theta, fatAbsorb, rank);
  nvtxRangePop();

  CloseSliceFile(sPtr);
  #ifdef UNIFIED
    cudaFree(vpz);
    cudaFree(vsv);
    cudaFree(epsilon);
    cudaFree(delta);
    cudaFree(phi);
    cudaFree(theta);
    cudaFree(ch1dxx);
    cudaFree(ch1dyy);
    cudaFree(ch1dzz);
    cudaFree(ch1dxy);
    cudaFree(ch1dyz);
    cudaFree(ch1dxz);
    cudaFree(v2px);
    cudaFree(v2pz);
    cudaFree(v2sz);
    cudaFree(v2pn);
    cudaFree(pp);
    cudaFree(pc);
    cudaFree(qp);
    cudaFree(qc);
    cudaFree(fatAbsorb);
  #else
    free(vpz);
    free(vsv);
    free(epsilon);
    free(delta);
    free(phi);
    free(theta);
    free(ch1dxx);
    free(ch1dyy);
    free(ch1dzz);
    free(ch1dxy);
    free(ch1dyz);
    free(ch1dxz);
    free(v2px);
    free(v2pz);
    free(v2sz);
    free(v2pn);
    free(pp);
    free(pc);
    free(qp);
    free(qc);
    free(fatAbsorb);
  #endif

  exit(0);    
}