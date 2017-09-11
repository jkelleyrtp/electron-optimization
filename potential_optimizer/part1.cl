#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef cl_clang_storage_class_specifiers
	#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif


#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

inline double4 get_accel(double4 pos, double4 v, float4 coil, int num_coils, __global float* ee,__global float* ek) //where factor equals charge/mass
{
	//float4 coils[2] = {coil1, coil2};
	//float M_PI = 3.14159265358979323846f;

	float4 coils[1] = {coil};
		
		
	float E_k = 0.0f;
	float K_k = 0.0f;
	
	float4 b = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	float r = sqrt(pos.x*pos.x + pos.y*pos.y);
	float z = 0.0f;
	float B_r;

	float a;
	float B;
	float gamma_;
	float Q;
	int k;

    for( int i = 0; i < num_coils; i++ ){
		z = (pos.z - coils[i].x);
		// z, radius, b0, rotation int

		a = r/coils[i].y;
		B = z/coils[i].y;
		gamma_ = z/r;


		Q = ((1.0f + a) * (1.0f + a) + B*B);
		k = (int)(4.0f * a / Q *10000000);

		E_k = ee[k];
		K_k = ek[k];


		B_r = coils[i].z * (gamma_)/( M_PI * sqrt(Q)) * ( ( E_k * (1.0f + a*a + B*B) / (Q-4.0f*a) ) - K_k);

		b.z += coils[i].z * (1.0f)/( M_PI * sqrt(Q)) * (( E_k * (1.0f - a*a - B*B)/(Q-4.0f*a)) + K_k);
		b.x += pos.x / r * B_r;
		b.y += pos.y / r * B_r;

//		b.w = coils[i].z * (1.0f)/( M_PI * sqrt(Q));
	}

	return (double4)( v.y * b.z - v.z * b.y, v.z * b.x - v.x * b.z, v.x * b.y - v.y * b.x, 0.0f) * -175882002272.0;
}



__kernel void compute_trajectory(
	__global double4* positions,			//xyz, charge/mass
	__global double4* velocities,		//xyz
//	__global float4* all_coils,
	__global float* ee_tab,
	__global float* ek_tab,
	__global float4* dest,
  	 int num_particles,
  	 int num_steps,
	 int num_coils,
	 double dt,
	 int iter_nth

){
	unsigned int thread = get_global_id(0);

	double4 pos = positions[thread];
	double4 velo = velocities[thread];

	double4 accel;

//	float4 coils[2];
	
	float4 coils = (float4)(1.0f,1.0f,1.0f,1.0f);

	//double4 k1, k2, k3, k4, l1, l2, l3, l4;

	for(int iter = 0; iter<(num_steps); iter++){
		// Euler's Method

		for (int sub_int = 0; sub_int < iter_nth; sub_int++){

			//Runge Kutta 4th Order
			/*
			k1 = dt * get_accel(pos, velo, coil1, coil2, ee_tab, ek_tab);
			l1 = dt * velo;

			k2 = dt * get_accel(  (pos + (0.5f * l1)), velo, coil1, coil2, ee_tab, ek_tab);
			l2 = dt * (velo + (0.5f * k1));

			k3 = dt * get_accel( (pos + (0.5f * l2)), velo, coil1, coil2, ee_tab, ek_tab);
			l3 = dt * (velo + (0.5f * k2));

			k4 = dt * get_accel( (pos + l3), velo, coil1, coil2, ee_tab, ek_tab);
			l4 = dt * (velo + k3);

			velo += (k1 + (2.0f*k2) + (2.0f*k3) +k4)/6.0f;
			pos += (l1 + (2.0f*l2) + (2.0f*l3) +l4)/6.0f;
			*/


			accel = get_accel(pos, velo, coils, num_coils, ee_tab, ek_tab); //returns acceleration
			velo += (accel * dt);
			pos += (velo * dt);



		}
		dest[thread*num_steps + iter] = (float4)(pos.x, pos.y, pos.z, 0.0f);
	}
}

/*
Device Info:

1 gb Max Global Memory
32 kbytes local memory
64 kbytes constant memory
6 compute units
657 MHz
float4s favored



*/
