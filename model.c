#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"


void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, 
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, 
           float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz, 
           float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
           float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
	   float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
	   float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
	   float * restrict phi, float * restrict theta, float * restrict fatAbsorb, int rank)
{

  float tSim=0.0;
  int nOut=1;
  float tOut=nOut*dtOutput;

  const long totalsamples=((long)sx*sy)*((long)sz*st);


  // DRIVER_Initialize initialize target, allocate data etc
  DRIVER_Initialize(  rank,   sx,   sy,   sz,   bord,
		      dx,  dy,  dz,  dt,
		      ch1dxx,    ch1dyy,    ch1dzz, 
		      ch1dxy,    ch1dyz,    ch1dxz, 
		      v2px,    v2pz,    v2sz,    v2pn,
		      vpz,    vsv,    epsilon,    delta,
		      phi,    theta,    fatAbsorb,
		      pp,    pc,    qp,    qc);

  
  double walltime=0.0;
  for (int it=1; it<=st; it++) {

    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    
    DRIVER_InsertSource(dt,it-1,iSource,pc,qc,src);

    const double t0=wtime();
    DRIVER_Propagate(  sx,   sy,   sz,   bord,
                 dx,   dy,   dz,   dt,   it,
                  ch1dxx,    ch1dyy,    ch1dzz,
                  ch1dxy,    ch1dyz,    ch1dxz,
                  v2px,    v2pz,    v2sz,    v2pn,
                  pp,    pc,    qp,    qc);

    SwapArrays(&pp, &pc, &qp, &qc);
    
    walltime+=wtime()-t0;

#if ((defined _ABSOR_SPHERE) || (defined _ABSOR_SQUARE))
    AbsorbingBoundary(sx, sy, sz, fatAbsorb, pc, qc);
#endif

    tSim=it*dt;
    if (tSim >= tOut) {

      DRIVER_Update_pointers(sx,sy,sz,pc);
      DumpSliceFile(sx,sy,sz,pc,sPtr);
      tOut=(++nOut)*dtOutput;
#ifdef _DUMP
      DRIVER_Update_pointers(sx,sy,sz,pc);
      DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
#endif
    }
  }
  printf ("MSamples/s %.0lf\n", 1.0e-6*totalsamples/walltime);

  // DRIVER_Finalize deallocate data, clean-up things etc 
  DRIVER_Finalize();

}

