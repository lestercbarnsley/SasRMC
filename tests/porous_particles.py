#%%
# from dataclasses import dataclass
# from typing import List

import numpy as np
from matplotlib import pyplot as plt

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


RADIUS = 68.0 / 10000 # ANGSTROM
RADIUS_POLYD = 0.1# 10%
LATTICE_PARAM = 92.0
HEIGHT = 0.4e4# is a micron#100e3#300000.0 ### Make this extremely long?
SOLVENT_SLD = 8.29179504046
PARTICLE_NUMBER = 50#0
CYCLES = 30

def create_box(particle_number: int, particle_factory: ParticleFactory ) -> Box:
    particles = [particle_factory.create_particle() for _ in range(particle_number)]
    cube = Cube(dimension_0=HEIGHT, dimension_1=HEIGHT, dimension_2=HEIGHT)
    i_total = int(np.sqrt(particle_number)) + 1
    for i in range(i_total):
        for j in range(i_total):
            ij = i * i_total+j
            if ij < len(particles):
                particle = particles[ij]
                particles[ij] = particle.set_position(Vector(1 * i * LATTICE_PARAM, 1 * j * LATTICE_PARAM))
    
    return Box(particles=particles, cube = cube)

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
    box_list = [box]
    print('starting config collision found',box.collision_test())
    data_src = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\20220204113917_p6mm_normalized_and_merged.txt"
    data = np.genfromtxt(data_src, delimiter='\t' )
    q = data[:,0]
    q_limited = lambda arr : np.array([a for a, q_i in zip(arr, q) if q_i > 2e-2])
    intensity = data[:,1]
    intensity_err = data[:,2]
    r_array = np.linspace(0, HEIGHT, num = 100)
    single_profile_calculator = ProfileCalculatorAnalytical(q_limited(q))#ProfileCalculator(q_array=q, r_array=r_array)
    profile_fitter = ProfileFitter(box_list=box_list, single_profile_calculator=single_profile_calculator, experimental_intensity=q_limited(intensity), intensity_uncertainty=q_limited(intensity_err))
    simulation = ScatteringSimulation(profile_fitter, simulation_params=sas_rmc.box_simulation_params_factory())
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



if __name__ == "__main__":
    runner = create_runner()
    #runner.run()
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
    plt.loglog(q, fitter.experimental_intensity, 'b.')
    rescale = simulation.simulation_params.get_value(key = constants.NUCLEAR_RESCALE)
    plt.loglog(q, fitter.simulated_intensity(rescale = rescale), 'r-')
    plt.show()


#%%
