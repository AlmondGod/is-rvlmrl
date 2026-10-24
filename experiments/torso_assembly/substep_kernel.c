#include <mujoco/mujoco.h>
#include <string.h>
#include "connector_kernel.c"
/* CPU fusion of the original internal-step loop. No fast-math. */
typedef char require_double_mjtNum[(sizeof(mjtNum)==sizeof(double))?1:-1];
static void frame(const mjModel *m,mjData *d,int body,double *f){
 double vel[6],r[3],v[3];
 mj_objectVelocity(m,d,mjOBJ_BODY,body,vel,0);
 for(int i=0;i<3;i++){f[i]=d->xpos[3*body+i];r[i]=f[i]-d->xipos[3*body+i];}
 memcpy(f+3,d->xmat+9*body,9*sizeof(double));
 cross3(vel,r,v);
 for(int i=0;i<3;i++){f[12+i]=vel[3+i]+v[i];f[15+i]=vel[i];}
}
static void audit(mjData *d,double *state,int *geoms){
 for(int i=0;i<d->ncon;i++){
  double depth=-d->contact[i].dist;
  if(depth>state[0]){state[0]=depth;state[1]=d->time;geoms[0]=d->contact[i].geom1;geoms[1]=d->contact[i].geom2;}
 }
}
int substeps(const mjModel *m,mjData *d,int steps,int nbody,const int *bodies,
 int n,const int *indices,const double *local_axis,double *inputs,
 const int *actuators,const double *limits,int nact,int panel_dof,
 double *state,int *geoms,double *failure_inputs,double *failure_times,int *failure_order){
 if(nbody>16||n>8||n<0||nbody<0)return 1;
 for(int i=0;i<6*m->nbody;i++)if(d->xfrc_applied[i]!=0.)return 2;
 for(int i=0;i<m->nv;i++)if(d->qfrc_applied[i]!=0.)return 3;
 double frames[16*18],output[13],zero[3]={0.,0.,0.},negative[3];
 int failures=0;
 for(int step=0;step<steps;step++){
  memset(d->qfrc_applied,0,m->nv*sizeof(mjtNum));
  for(int j=0;j<nbody;j++)frame(m,d,bodies[j],frames+18*j);
  for(int j=0;j<n;j++){
   double *x=inputs+56*j;int ai=indices[2*j],bi=indices[2*j+1];
   if(ai<0||ai>=nbody||bi<0||bi>=nbody)return 4;
   memcpy(x,frames+18*ai,18*sizeof(double));memcpy(x+18,frames+18*bi,18*sizeof(double));
   const double *axis=local_axis+3*j;
   for(int k=0;k<3;k++)x[42+k]=fma(x[3+3*k+2],axis[2],fma(x[3+3*k],axis[0],x[3+3*k+1]*axis[1]));
   connectors(x,output,1);
   if(x[51]==0.&&output[8]!=0.){
    memcpy(failure_inputs+56*failures,x,56*sizeof(double));
    failure_times[failures]=d->time;failure_order[failures++]=j;
   }
   memcpy(x+51,output+8,5*sizeof(double));state[3]+=output[7];
   mj_applyFT(m,d,output,zero,output+3,bodies[ai],d->qfrc_applied);
   for(int k=0;k<3;k++)negative[k]=-output[k];
   mj_applyFT(m,d,negative,zero,output+3,bodies[bi],d->qfrc_applied);
  }
  if(panel_dof>=0)state[4]=d->qfrc_applied[panel_dof];
  mj_step(m,d);audit(d,state,geoms);
  for(int k=0;k<nact;k++)if(fabs(d->actuator_force[actuators[k]])>=limits[k]){state[2]+=1.;break;}
 }
 mj_forward(m,d);audit(d,state,geoms);
 memset(d->qfrc_applied,0,m->nv*sizeof(mjtNum));
 state[5]=(double)failures;
 return 0;
}
