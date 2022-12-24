#%%
# from dataclasses import dataclass
# from typing import List

from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import fixed_quad
from scipy.optimize import curve_fit

import sas_rmc
from sas_rmc import constants, Vector
from sas_rmc.factories.controller_factory import ControllerFactory
from sas_rmc.factories.acceptable_command_factory import MetropolisAcceptanceFactory
from sas_rmc.factories.annealing_config import VeryFastAnneal
from sas_rmc.factories.particle_factory_cylindrical import CylindricalParticleFactory, CylindricalCommandFactory, ParticleFactory, CylindricalLongParticleFactory
from sas_rmc.factories.simulator_factory import MemorizedSimulatorFactory   
from sas_rmc.box_simulation import Box, Cube
from sas_rmc.profile_calculator import ProfileFitter
from sas_rmc.result_calculator import ProfileCalculator, ProfileCalculatorAnalytical
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.logger import Logger, LogCallback

rng = constants.RNG
PI = constants.PI


RADIUS = 68.0 /3 # ANGSTROM
RADIUS_POLYD = 0.3# 10%
LATTICE_PARAM = 4 * PI / (np.sqrt(3) *  0.0707 )  #100#    / (np.sqrt(3) / 2)#92.0
print(LATTICE_PARAM)
HEIGHT = 0.4e4# is a micron#100e3#300000.0 ### Make this extremely long?
SOLVENT_SLD = 8.29179504046
PARTICLE_NUMBER =200#0
CYCLES = 200 
RUN = True
STARTING_RESCALE = 1

TWOPIDIVQMIN = 2 * PI / 0.002

def create_box(particle_number: int, particle_factory: ParticleFactory ) -> Box:
    particles = [particle_factory.create_particle() for _ in range(particle_number)]
    cube = Cube(dimension_0=TWOPIDIVQMIN, dimension_1=TWOPIDIVQMIN, dimension_2=HEIGHT)
    i_total = int(np.sqrt(particle_number)) + 1
    for i in range(i_total):
        for j in range(i_total):
            ij = i * i_total+j
            if ij < len(particles):
                particle = particles[ij]
                particles[ij] = particle.set_position(Vector(1 * i * LATTICE_PARAM, 1 * j * LATTICE_PARAM))
    
    box= Box(particles=particles, cube = cube)
    #box.force_inside_box(in_plane= True)
    return box

def create_line(particle_number: int, particle_factory: ParticleFactory ) -> Box:
    particles = [particle_factory.create_particle().set_position(Vector(i * LATTICE_PARAM, 0)) for i in range(particle_number)]
    return Box(particles=particles, cube=Cube(dimension_0=HEIGHT, dimension_1=HEIGHT))

def create_hexagonal_box(particle_number: int, particle_factory: ParticleFactory) -> Box:
    ROOT_THREE_ON_TWO = np.sqrt(3) / 2
    i_total = int(np.sqrt(particle_number)) + 1
    particles = []
    starting_x = -int(i_total / 2) * LATTICE_PARAM
    starting_y = -int(i_total) / 2 * ROOT_THREE_ON_TWO * LATTICE_PARAM
    for j in range(i_total):
        starting_x_this_row = starting_x if j % 2 == 0 else (starting_x + (1/2) * LATTICE_PARAM)
        y_pos = starting_y + j * ROOT_THREE_ON_TWO * LATTICE_PARAM
        for i in range(i_total):
            x_pos = starting_x_this_row + i * LATTICE_PARAM
            particles.append(
                particle_factory.create_particle().set_position(Vector(x_pos, y_pos))
                )
    size_x = 6* max(particle.shapes[0].radius for particle in particles) + max(particle.position.x for particle in particles) - min(particle.position.x for particle in particles)
    size_y= 6* max(particle.shapes[0].radius for particle in particles) + max(particle.position.y for particle in particles) - min(particle.position.y for particle in particles)
    return Box(particles = particles, cube = Cube(dimension_0=size_x, dimension_1=size_y, dimension_2=HEIGHT + 0.1))
            
    
    

def power_law(q, power, forward_intensity):
    return forward_intensity * (q**power)

def create_runner() -> RmcRunner:
    
    particle_factory = CylindricalLongParticleFactory(
        command_factory=CylindricalCommandFactory(nominal_step_size=RADIUS),
        cylinder_radius=RADIUS,
        cylinder_radius_polydispersity=RADIUS_POLYD,
        cylinder_height=HEIGHT,
        cylinder_height_polydispersity=0.0,
        cylinder_sld=0.0,
        solvent_sld=SOLVENT_SLD
    )
    box = create_box(PARTICLE_NUMBER, particle_factory=particle_factory)
    box = create_hexagonal_box(PARTICLE_NUMBER, particle_factory)
    '''box = Box(particles = [particle_factory.create_particle() for _ in range(PARTICLE_NUMBER)], cube = Cube(dimension_0=TWOPIDIVQMIN, dimension_1=TWOPIDIVQMIN, dimension_2=HEIGHT))
    box.force_inside_box(in_plane=True)
    for i, particle in enumerate(box.particles):
        box.particles[i] = particle.set_orientation(Vector(0,0,1))'''
    #box = create_line(PARTICLE_NUMBER, particle_factory)
    box_list = [box]
    print('starting config collision found',box.collision_test())
    data_src = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\20220204113917_p6mm_normalized_and_merged.txt"
    data = np.genfromtxt(data_src, delimiter='\t' )
    nominal_q_min = 2 * PI / np.min([box.cube.dimension_0, box.cube.dimension_1])
    nominal_q_min = 0.02
    q = data[:,0]
    q_limited = lambda arr : np.array([a for a, q_i in zip(arr, q) if q_i > nominal_q_min])
    #q_limited = lambda arr: arr
    intensity = data[:,1]
    intensity_err = data[:,2]
    r_array = np.linspace(0, HEIGHT, num = 100)
    single_profile_calculator = ProfileCalculatorAnalytical(q_limited(q))#ProfileCalculator(q_array=q, r_array=r_array)
    profile_fitter = ProfileFitter(box_list=box_list, single_profile_calculator=single_profile_calculator, experimental_intensity=q_limited(intensity), intensity_uncertainty=q_limited(intensity_err))
    simulation = ScatteringSimulation(profile_fitter, simulation_params=sas_rmc.box_simulation_params_factory(STARTING_RESCALE))
    controller_factory = ControllerFactory(
        annealing_config=VeryFastAnneal(annealing_stop_cycle_number=CYCLES / 2,anneal_start_temp=10.0, anneal_fall_rate=0.1),
        total_cycles=CYCLES,
        p_factory=particle_factory,
        acceptable_command_factory=MetropolisAcceptanceFactory()
        )
    controller = controller_factory.create_controller(simulation.simulation_params, box_list)
    simulator_factory = MemorizedSimulatorFactory(controller, simulation, box_list)
    simulator = simulator_factory.create_simulator()
    plt.plot([p.position.x for p in box.particles],[p.position.y for p in box.particles], 'b.')
    plt.show()
    plt.loglog(q, intensity, 'b-')
    
    '''from scipy.optimize import curve_fit

    popt, _ = curve_fit(power_law, q[:30], intensity[:30], p0 = [-4, intensity[0]])
    plt.loglog(q[:30], power_law(q[:30], *popt), 'r-')
    plt.loglog(q, power_law(q, -4, popt[1]), 'g-')
    plt.show()
    print(popt)

    plt.loglog(q, intensity -power_law(q, -4, popt[1]), 'r-')'''
    plt.show()
    

    return RmcRunner(
        logger = Logger(callback_list=[]),
        simulator=simulator,
        force_log=True)



def i_of_q(q: np.ndarray, intensity: np.ndarray, q_i: float):
    argmin = np.argmin((q-q_i)**2)
    if argmin < 5:
        popt, _ = curve_fit(power_law, q[:5], intensity[:5], p0 = [-4, intensity[0]])
        return power_law(q_i, *popt)
    if argmin > q.shape[0] - 5:
        popt, _ = curve_fit(power_law, q[-5:], intensity[-5:], p0 = [-4, intensity[-5]])
        return power_law(q_i, *popt)
    interpolator = interp1d(q[argmin - 2: argmin + 3], intensity[argmin - 2: argmin + 3])
    return interpolator(q_i)


    

def pr_inversion(q:np.ndarray, intensity:np.ndarray):
    r = np.linspace(2 * PI / np.max(q), 2 * PI / np.min(q), num = q.shape[0])
    inversion = lambda q_i, r_i : np.array([i_of_q(q, intensity, q_i_i) * np.sinc(q_i_i * r_i / PI) * q_i_i**2 for q_i_i in q_i]) 
    return r, (r**2 / (2*PI**2)) * np.array([fixed_quad(lambda q_i : inversion(q_i, r_i), 1e-4, 1e4)[0] for r_i in r])
    
    




if __name__ == "__main__":
    runner = create_runner()
    if RUN:
        runner.run()
    simulation = runner.simulator.evaluator.simulation
    fitter = simulation.fitter
    box = fitter.box_list[0]
    print('ending config collision found',box.collision_test())
    print(min((p_1.position - p_2.position).mag for p_1 in box.particles for p_2 in box.particles if p_1 is not p_2))
    plt.plot([p.position.x for p in box.particles],[p.position.y for p in box.particles], 'b.')
    plt.show()
    plt.hist([(p_1.position - p_2.position).mag for p_1 in box.particles for p_2 in box.particles], bins = 100)
    plt.show()
    q = fitter.single_profile_calculator.q_array
    #r, pr = pr_inversion(q, fitter.experimental_intensity - 0.01)
    #plt.plot(r, pr)
    #plt.show()
    plt.loglog(q, fitter.experimental_intensity, 'b.')
    rescale = simulation.simulation_params.get_value(key = constants.NUCLEAR_RESCALE)
    plt.loglog(q, fitter.simulated_intensity(rescale = rescale), 'r-')
    plt.show()


#%%
