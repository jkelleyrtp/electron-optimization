#include <stdio.h>
#include <math.h>
// calculated mutual inductance of two parallel coils,
// 1 meter in diameter, 1 meter apart, 1 turn
#define nseg 10
#define pi 3.1415925

typedef struct {
double x;
double y;
double z;
} vector;

double dot(vector a, vector b){
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}

double mu_prime(vector sensor_coord, vector sensor_tangent){
  double theta; // angle along path, cylindrical coordinates
  double dtheta; // differential angle along path
  double r=0.5; // radius of loop = 1 m diameter
  double z_source= 0.5; // position of source loop
  double z_sensor=-0.5; // position of sensor loop
  double s; // path length along coil
  double ds; // differential path length
  double mag_deltar; // difference between source and sensor
  double mu; // mu_prime
  double d_mu; // incremental mu_prime
  vector source_coord, source_tangent,deltar;
  int nturns = 1; // number of turns
  long i,j;
  s = 0.0;
  mu = 0.0;
  for (i=0; i<nseg*nturns; i++){ // integrate along source path
    dtheta = 2.0*pi/nseg;
    theta = dtheta*i;
    ds = r*dtheta;
    s += ds;
    // do the contribution from inner radius of trace
    source_coord.x = r*cos(theta);
    source_coord.y = r*sin(theta);
    source_coord.z = 0.5; // plane
    source_tangent.x = -sin(theta);
    source_tangent.y = cos(theta);
    source_tangent.z = 0;

    deltar.x = source_coord.x - sensor_coord.x;
    deltar.y = source_coord.y - sensor_coord.y;
    deltar.z = source_coord.z - sensor_coord.z;
    mag_deltar = sqrt(dot(deltar,deltar));
    d_mu = dot(source_tangent, sensor_tangent)/mag_deltar;
    mu += d_mu*ds;
  }
return (mu);
}


double mu(void) // get mutual inductance
{
  double theta; // angle along path, cylindrical coordinates
  double dtheta; // differential angle along path
  double r=0.5; // radius of loop = 1 m diameter
  double z_source= 0.5; // position of source loop
  double z_sensor=-0.5; // position of sensor loop
  double s; // path length along coil
  double ds; // differential path length
  double mag_deltar; // difference between source and sensor
  double mu; // mu_prime
  double d_mu; // incremental mu_prime
  vector source_coord, source_tangent,deltar;
  vector sensor_coord, sensor_tangent;
  int nturns = 1; // number of turns
  long i,j;
  s = 0.0;
  mu = 0.0;

  for (i=0; i<nseg*nturns; i++) { // integrate along sensor path
    dtheta = 2.0*pi/nseg;
    theta = dtheta*i;
    r = 0.5;
    sensor_coord.x = r*cos(theta);
    sensor_coord.y = r*sin(theta);
    sensor_coord.z = -0.5; // plane
    sensor_tangent.x = -sin(theta);
    sensor_tangent.y = cos(theta);

    sensor_tangent.z = 0;
    printf("."); if((i%80) == 0) printf("\n"); // activity
    ds = r*dtheta;
    mu += ds*mu_prime(sensor_coord, sensor_tangent);
    s += ds;
  }

  printf("\ns = %g\n",s);
  mu *= 1.0e-7; // scale by mu_0/(4 pi)
  return (mu);
}


int main(void) {
  double M;
  M = mu();
  printf("M = %e\n\n",M);

  double L,R,l,a,b,beta,mu,K,E;
  L = 0.058;
  R = 0.058;
  l = 0.116;


  a = (L*L + R*R + l*l)/(L*L*R*R);
  b = 2.0/(L*R);
  beta = sqrt(2.0*b/(a+b));
  mu = 4.0e-7*pi;
  K = 1.8541;
  E = 1.3506;
  M = 2.0*mu*(sqrt(a+b)/b)*((1.0 - 0.5*beta*beta)*K - E);
  printf("a = %f\n",a);
  printf("b = %f\n",b);
  printf("beta = %f\n",beta);
  printf("M = %e\n\n",M);
}
