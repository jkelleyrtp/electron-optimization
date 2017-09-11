'''
True Workflow
 - Choose parameter to alter:
   - r
   - B_0
   - sep_dist

 - Choose start and stop ranges for each parameter
 - Initialize positions and velocities vectors from monte carlo style governor
 - Setup coil parameters for run set - No





structure
---------
inputs ->
  coil parameters
  electron gun parameters

outputs ->
  best coil distance for a given radius
  do coil radius and field strength affect each other?
  Idea is to input design restrictions and find best coil spacing
    --> see effects of increasing coil current (therefore B)
  --> total metric is minimizing average r^2 = x^2 + y^2 + (z-sep/2)^2
  -- bit of statistical analysis to determine how the curve is shifted (mean vs median)
  --> starting r vs avg r^2 distribution
  --> arange from 0 to r
  --> tweak

where better = lowest average r^2

Our goal is to produce a model and determine relationships.

for each coil arangement, test distance of l vs r, keeping D constant
--> hypothesis: larger L values will mean better runs
--> hypothesis: certain best range of coil sep values
--> hypothesis: field strength and coil radius will produce a ratio constant for all variations
  --> test by 2d array of coil radius x field strength
  --> heatmap graphic
--> insertion ratio

new_simulation


'''
# Imports
import sys
from math import sin, cos, tan, radians, sqrt
import pyopencl as cl
import numpy as np
import pyopencl.array as cl_array
from scipy.special import ellipk, ellipe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
mu_0 = 1.25663706e-6
ellipe_table = ellipe(np.arange(0,1, 1.0/10000000.0))
ellipk_table = ellipk(np.arange(0,1,1.0/10000000.0))


class all:
    def __init__(self):
        print '-- New all object created --'

        # call GPU building
            # initialize GPU
            # load single particle simulation code
            # pass positions, velocities, coils
            # electron gun function returns positions and velocities


    class _GPU:
        def __init__(self, filename, device_id = 1):
            # Setup OpenCL platform
            platform = cl.get_platforms()
            computes = [platform[0].get_devices()[device_id]]
            print "New context created on", computes
            self.ctx = cl.Context(devices=computes)
            self.queue = cl.CommandQueue(self.ctx)
            self.mf = cl.mem_flags

            # Open and build cl code
            f = open(filename, 'r')
            fstr = "".join(f.readlines())
            self.program = cl.Program(self.ctx, fstr).build()

        def execute(self, sim):
            # 1 float is 4 bytes

            # Clean buffers to conserve memory if performing multiple runs
            try:
                self.d_buf.release()
            except:
                print 'No buffers to clean'
            self.queue.finish()


            # Prepare input, output, and lookup val buffers
            self.p_buf = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.positions )        # Positions
            self.v_buf = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.velocities )       # Velocities
            self.coil_buf = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.coils )         # Coils
            self.ee = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.ee_table )            # Elliptical Integral 1
            self.ek = cl.Buffer(self.ctx,  self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=sim.ek_table )            # Elliptical Integral 2
            self.d_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, sim.bytesize * sim.num_particles * sim.num_steps)                 # Output r^2 buffer
            self.queue.finish()

            # Run Kernel
            kernelargs = (self.p_buf, self.v_buf, self.coil_buf, self.ee, self.ek, self.d_buf, sim.num_particles, sim.num_steps, np.int32(2), sim.dt, sim.iter_nth)

            print "Values successfully passed"

            self.program.compute_trajectory(self.queue, (int(sim.num_particles),), None, *(kernelargs))

            print "Kernels started"

            self.queue.finish()

            # Dump, clean, return -- must reshape data when using float4s
            self.ret_val = np.empty_like(np.ndarray((sim.num_particles, sim.num_steps, sim.bytesize/4)).astype(np.float32))
            read = cl.enqueue_copy(self.queue, self.ret_val, self.d_buf)
            self.queue.finish()
            read.wait()
#            print (read.profile.end-read.profile.start)
            self.d_buf.release()
            print "\a"
            return self.ret_val



    class _SIMOBJECT:
        def __init__(self, positions, velocities, coils, num_particles, steps, bytesize=4, iter_nth = 1, dt = .0000000000002 ):
            self.positions = positions.astype(np.float64)
            self.velocities = velocities.astype(np.float64)
            self.coils = np.array(coils).astype(np.float32)
            self.num_particles = np.int32(num_particles)
            self.num_steps = np.int32(steps)
            self.bytesize = bytesize
            self.ee_table = ellipe_table.astype(np.float32)
            self.ek_table = ellipk_table.astype(np.float32)

            self.dt = np.float64(dt)
            self.iter_nth = np.int32(iter_nth)



    class _COIL:
        def __init__(self, radius = 0.035, current = 10000, z_pos = 0.0):
            self.radius = radius
            self.current = current
            self.z_pos = z_pos
            self.position = [0.0, 0.0, z_pos, 0.0]
            self.B_0 = self.current * mu_0 / (2.0 * self.radius)
            self.arr = np.array([z_pos, radius, self.B_0, 0.0]).astype(np.float32)




    '''
    Takes array of coils and displays to screen. First and second coils are bounding
    box coils.

    Positions is list of positions
    '''
    def graph_trajectory(self, coils, positions):
        coil_1 = coils[0]
        coil_2 = coils[1]
        steps = len(positions)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:,0], positions[:,1], zs= positions[:,2])
        ax.set_xlim([coil_1.radius,coil_1.radius * -1])
        ax.set_ylim([coil_1.radius,coil_1.radius * -1])
        ax.set_zlim([0, coil_2.z_pos])

        theta = np.linspace(0, 2*np.pi, 100)

        # the radius of the circle
        r = coil_1.radius

        # compute x1 and x2

        loop_x = r*np.cos(theta)
        loop_y = r*np.sin(theta)
        loop_z=0
        ax.plot(loop_x,loop_y, loop_z)
        ax.plot(loop_x,loop_y, coil_2.z_pos)

        ax.scatter(positions[0][0],positions[0][1],positions[0][2], color="green")

        ax.scatter(positions[steps-2][0],positions[steps-2][1],positions[steps-2][2], color="red")


    def get_conf_times(self, r_vals, radius, z_pos, dt, iter_nth):
        conf_times = []

        for p in range(len(r_vals)) :
            x_conf = len(np.where( abs(r_vals[p][:,0]) < radius)[0]) * dt * iter_nth * 1e9
            y_conf = len(np.where( abs(r_vals[p][:,1]) < radius)[0]) * dt * iter_nth * 1e9
            z_conf = len(np.where( abs((z_pos/2.0) - r_vals[p][:,2]) < (z_pos/2.0))[0]) * dt * iter_nth * 1e9

            conf_times.append(np.amin([x_conf,y_conf,z_conf]))

        return conf_times



    def single_sim(self, device_id = 0):
        # Generate a single electron pos data
        # best of 1105.824 at -5000, 5000 [ 0,0.0004, -.03], [0,-5e3, 7e5]

        coil_1 = self._COIL( radius = .05, current = -7150, z_pos = 0.0 )
        coil_2 = self._COIL( radius = .05, current = 7150, z_pos = 0.055 )
        coils = [coil_1.arr, coil_2.arr]

        # Constants
        e_charge = 1.6e-19  # Coulombs
        e_mass = 9.1e-31    # Kilograms
        e_gun_energy = 1000 # measured in volts
        avg_velo = sqrt( (2.0 * e_gun_energy * e_charge) / e_mass) # m/s

        positions = np.array([[0.000 ,0.001, -0.03, 0.0,]])
        velocities = np.array([[0.0, 0, .5006e7 ,0.0,]])
        print velocities

        num_particles = 1
        steps = 35000; #350000;
        bytesize = 16
        iter_nth = 36;
        dt = .0000000000002

        self.SINGLE_SIM = self._SIMOBJECT(positions, velocities, coils, num_particles, steps, bytesize = bytesize, iter_nth=iter_nth, dt = dt)
        self.SINGLE_SIM.calculator = self._GPU(path_to_integrator, device_id)

        self.SINGLE_SIM.r_vals = self.SINGLE_SIM.calculator.execute( self.SINGLE_SIM)

        a = self.SINGLE_SIM.r_vals[0]

        self.graph_trajectory([coil_1, coil_2], a)

        self.SINGLE_SIM.conf_times = self.get_conf_times(self.SINGLE_SIM.r_vals, coil_1.radius, coil_2.z_pos, dt, iter_nth)
        #self, r_vals, radius, z_pos, dt, iter_nth

        print "Total confinement:", self.SINGLE_SIM.conf_times[0]
        plt.title(("Total confinement:", self.SINGLE_SIM.conf_times[0], " ns"))
        plt.show()

    def dim_by_dim(self, device_id = 0):
        '''
        The purpose of this simulation is to manipulate two selected parameters
        against eachother with 10,000 simualted particles per run (or less).

        Options of parameters include
        * Coil Current	          - \
        * Coil radius             - / overall b0 - similar b0s mean more current at higher rs but likely better capture time
        * Coil Separation       --- good graphics on B over Z and comparing cusp region shapes

        * Electron beam KeV
        * Electron beam angle (>1.0 deg, and no angle)
        '''

        # Constants
        e_charge = 1.6e-19  # Coulombs
        e_mass = 9.1e-31    # Kilograms

        # Parameters
        e_gun_energy = 5 # measured in volts
        avg_velo = sqrt( (2.0 * e_gun_energy * e_charge) / e_mass) # m/s


        # Control parameters
        memory = 3000000000 # bytes
        bytesize = 16

        slices = 1000
        e_per_slice = 20
        num_particles = e_per_slice * slices
        total_steps = 7000000 # ten million


        mem_p_particle = memory/num_particles # can serve so many bytes to display
        steps = mem_p_particle/bytesize
        iter_nth = total_steps/steps
        print "Steps: ",steps," iter_nth: ", iter_nth


#        steps = 80000;
#        iter_nth = 180;
        dt = .0000000000002



        e_gun_z = -.03       # in meters
        e_gun_angle = .1  # in degrees


        coil_1 = self._COIL( radius = .05, current = -5350, z_pos = 0.0 )
        coil_2 = self._COIL( radius = .05, current = 5350, z_pos = 0.055 )
        coils = np.array([coil_1.arr, coil_2.arr])

        thetas = np.linspace(0.001, radians(e_gun_angle), e_per_slice)

        energies = np.linspace(1.0, avg_velo, slices)

        radii = np.linspace(0.0003, 0.001, slices)

        self.energies = energies
        self.thetas = thetas

        velo_p_slice = np.zeros((e_per_slice, 4))
#        velo_p_slice[:,2] = avg_velo * np.cos(thetas) # Set Z components
#        velo_p_slice[:,0] = avg_velo * np.sin(thetas) # Set X components
        velo_p_slice[:,2] = avg_velo

        self.velo_p_slice = velo_p_slice

        self.velocities = np.tile(z.velo_p_slice, (slices,1,1)).reshape(num_particles, 4)  #* energies.reshape(slices,1,1)
        velocities = self.velocities

        radii_array = np.repeat(radii, e_per_slice)
        self.positions = np.tile( [0.0, 0.0, e_gun_z, 0.0], (num_particles, 1))
        self.positions[:,1] = radii_array
        positions = self.positions


        self.DIM_SIM = self._SIMOBJECT(positions, velocities, coils, num_particles, steps, bytesize = bytesize, iter_nth=iter_nth, dt = dt)
        self.DIM_SIM.calculator = self._GPU(path_to_integrator, device_id = device_id)

        self.DIM_SIM.r_vals = self.DIM_SIM.calculator.execute( self.DIM_SIM)
        print "Simulation complete"

        #self.a = self.DIM_SIM.r_vals.reshape((num_particles,steps, 4))

        self.DIM_SIM.conf_times = self.get_conf_times(self.DIM_SIM.r_vals)

        self.conf_times = []

        for p in range(len(self.DIM_SIM.r_vals)) :
            x_conf = len(np.where( abs(self.DIM_SIM.r_vals[p][:,0]) < coil_1.radius)[0]) * dt * iter_nth * 1e9
            y_conf = len(np.where( abs(self.DIM_SIM.r_vals[p][:,1]) < coil_1.radius)[0]) * dt * iter_nth * 1e9
            z_conf = len(np.where( abs((coil_2.z_pos/2.0) - self.DIM_SIM.r_vals[p][:,2]) < (coil_2.z_pos/2.0))[0]) * dt * iter_nth * 1e9

            self.conf_times.append(np.amin([x_conf,y_conf,z_conf]))




    def multi_sim(self):

        coil_1 = self._COIL( radius = .05, current = -6000, z_pos = 0.0 )
        coil_2 = self._COIL( radius = .05, current = 6000, z_pos = 0.055 )
        coils = [coil_1.arr, coil_2.arr]

        # Control parameters
        memory = 3000000000 # bytes
        bytesize = 16

        num_particles = 10000
        total_steps = 9000000 # ten million
        dt = .0000000000002

        mem_p_particle = memory/num_particles # can serve so many bytes to display
        steps = mem_p_particle/bytesize
        iter_nth = total_steps/steps
        print "Steps: ",steps," iter_nth: ", iter_nth


        positions = np.tile( [0.000 ,0.0, -0.03, 0.0], (num_particles, 1))
        velocities = np.tile ([1e2 , 0.0 , .75e6 ,0.0],(num_particles, 1) )
        positions[:,1] = np.linspace(0.0, 0.01, num_particles)



        self.MULTI_SIM = self._SIMOBJECT(positions, velocities, coils, num_particles, steps, bytesize = bytesize, iter_nth=iter_nth, dt = dt)
        self.MULTI_SIM.calculator = self._GPU(path_to_integrator, device_id = 2)

        self.MULTI_SIM.r_vals = self.MULTI_SIM.calculator.execute( self.MULTI_SIM)

        a = self.MULTI_SIM.r_vals
        self.conf_times = []

        for p in range(len(z.MULTI_SIM.r_vals)) :
            x_conf = len(np.where( abs(self.MULTI_SIM.r_vals[p][:,0]) < coil_1.radius)[0]) * dt * iter_nth * 1e9
            y_conf = len(np.where( abs(self.MULTI_SIM.r_vals[p][:,1]) < coil_1.radius)[0]) * dt * iter_nth * 1e9
            z_conf = len(np.where( abs((coil_2.z_pos/2.0) - self.MULTI_SIM.r_vals[p][:,2]) < (coil_2.z_pos/2.0))[0]) * dt * iter_nth * 1e9

            self.conf_times.append(np.amin([x_conf,y_conf,z_conf]))


#path_to_integrator = '/Users/jonkelley/Desktop/temp_potentia/potential_optimizer/part1.cl'
#z.dim_by_dim()
#z.single_sim()
#z.EGUNvsDIST()
#z.single_sim()

import os
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
rel_path = "part1.cl"
path_to_integrator = os.path.join(script_dir, rel_path)

z = 0;

if __name__ == "__main__":
    z = all()
    simulations = {
        'single':z.single_sim,
        'main_sim':z.dim_by_dim
    }
    if len(sys.argv) == 1:
        print "single sim"
        z.single_sim(0)
    else:
        simulations[sys.argv[1]](int(sys.argv[2]))


        # %run potential_optimizer.py{'single'} {0}
