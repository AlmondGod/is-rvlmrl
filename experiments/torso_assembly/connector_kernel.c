#include <stddef.h>
#include <math.h>
/* Preserve the reference operation order. Vector dot products are unfused;
 * the supported NumPy matrix-vector kernel uses the explicit nested FMAs below.
 * The Python loader checks compatibility and refuses unsupported runtimes.
 * No fast-math or machine-specific compilation flags are permitted. */
static double dot3(const double *a,const double *b){return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2];}
static void cross3(const double *a,const double *b,double *o){o[0]=a[1]*b[2]-a[2]*b[1];o[1]=a[2]*b[0]-a[0]*b[2];o[2]=a[0]*b[1]-a[1]*b[0];}
static void world(const double *f,const double *anchor,double *out){for(int i=0;i<3;i++)out[i]=f[i]+fma(f[3+3*i+2],anchor[2],fma(f[3+3*i],anchor[0],f[3+3*i+1]*anchor[1]));}
static void velocity(const double *f,const double *point,double *out){double r[3],v[3];for(int i=0;i<3;i++)r[i]=point[i]-f[i];cross3(f+15,r,v);for(int i=0;i<3;i++)out[i]=f[12+i]+v[i];}
void connectors(const double *input,double *output,size_t n){
 for(size_t row=0;row<n;row++){
  const double *a=input+56*row,*b=a+18,*aa=a+36,*ab=a+39,*axis=a+42,*p=a+45,*s=a+51;
  double *o=output+13*row,pa[3],pb[3],point[3],delta[3],va[3],vb[3],v[3],force[3],t[3];
  world(a,aa,pa);world(b,ab,pb);
  for(int i=0;i<3;i++){point[i]=(pa[i]+pb[i])/2.;delta[i]=pb[i]-pa[i];o[i]=0.;o[3+i]=point[i];}
  o[6]=0.;o[7]=0.;for(int i=0;i<5;i++)o[8+i]=s[i];
  if(s[0]!=0.||p[0]<=0.)continue;
  velocity(a,point,va);velocity(b,point,vb);
  for(int i=0;i<3;i++){v[i]=vb[i]-va[i];force[i]=p[0]*delta[i]+p[1]*v[i];}
  double normal=dot3(force,axis);
  for(int i=0;i<3;i++)t[i]=force[i]-normal*axis[i];
  double shear=sqrt(dot3(t,t)),u=sqrt(pow(normal/p[2],2.)+pow(shear/p[3],2.));
  if(u>s[2])o[10]=u;
  double distance2=dot3(delta,delta),energy=.5*p[0]*distance2;
  o[11]=s[3]+.5*(p[0]-s[4])*distance2;o[12]=p[0];
  if(u>=1.||sqrt(distance2)>=p[4]){o[8]=1.;o[9]=s[1]+energy;continue;}
  for(int i=0;i<3;i++)o[i]=force[i];
  o[6]=energy;o[7]=p[1]*dot3(v,v)*p[5];
 }
}
