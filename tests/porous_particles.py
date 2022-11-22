#%%
from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib import pyplot as plt

import sas_rmc
from sas_rmc import constants, Vector
from sas_rmc.factories.controller_factory import ControllerFactory
from sas_rmc.factories.acceptable_command_factory import MetropolisAcceptanceFactory
from sas_rmc.factories.annealing_config import VeryFastAnneal
from sas_rmc.factories.particle_factory_cylindrical import CylindricalParticleFactory, CylindricalCommandFactory, ParticleFactory
from sas_rmc.factories.simulator_factory import MemorizedSimulatorFactory   
from sas_rmc.box_simulation import Box, Cube
from sas_rmc.profile_calculator import ProfileFitter
from sas_rmc.result_calculator import ProfileCalculator
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.logger import Logger, LogCallback

rng = constants.RNG
'''
def create_runner():
    detector_list = detector_builder.MultipleDetectorBuilder(dataframes, config_dict).build_detector_images()
    p_factory = dict_to_particle_factory(config_dict)
    cont_factory = controller_factory.gen_from_dict(config_dict, p_factory=p_factory, acceptable_command_factory=MetropolisAcceptanceFactory())
    boxfactory = box_factory.gen_from_dict(config_dict, detector_list)
    box_list_factory = box_factory.gen_list_factory_from_dict(config_dict)
    box_list = box_list_factory.create_box_list(boxfactory, p_factory)
    result_calculator_maker_factory = dict_to_result_maker_factory(config_dict, p_factory)
    sim_factory = simulation_factory.gen_from_dict(config_dict, result_calculator_maker_factory.create_result_maker())
    simulation = sim_factory.create_simulation(detector_list, box_list)
    controller = cont_factory.create_controller(simulation.simulation_params, box_list)
    simulator = simulator_factory.MemorizedSimulatorFactory(controller, simulation, box_list).create_simulator()
    # This should be a builder, not a factory
    save_path_maker = generate_file_path_maker(output_path, config_dict.get("simulation_title", ""))
    file_format = config_dict.get("output_plot_format", "PDF").lower()
    excel_callback = ExcelCallback(save_path_maker, box_list, detector_list, controller )
    detector_plotter = DetectorPlotter(save_path_maker, detector_list, format=file_format, make_initial=False)
    profile_plotter= ProfilePlotter(save_path_maker, detector_list, format=file_format, make_initial=False)
    box_plotter = BoxPlotter(save_path_maker, box_list, format = file_format, make_initial=False)
    return RmcRunner(
        logger=Logger(callback_list=[excel_callback, detector_plotter, profile_plotter, box_plotter]),
        simulator=simulator,
        force_log=config_dict.get("force_log_file", True)
    )

'''
'''@dataclass
class 
    '''


RADIUS = 68.0 # ANGSTROM
RADIUS_POLYD = 0.1 # 10%
LATTICE_PARAM = 92.0
HEIGHT = 3000.0
SOLVENT_SLD = 8.29179504046
PARTICLE_NUMBER = 100
CYCLES = 10

def create_box(particle_number: int, particle_factory: ParticleFactory ) -> Box:
    particles = [particle_factory.create_particle() for _ in range(particle_number)]
    cube = Cube(dimension_0=HEIGHT, dimension_1=HEIGHT, dimension_2=HEIGHT)
    i_total = int(np.sqrt(particle_number))
    for i in range(i_total):
        for j in range(i_total):
            ij = i * i_total+j
            if ij < len(particles):
                particle = particles[ij]
                particles[ij] = particle.set_position(Vector(2 * i * LATTICE_PARAM, 2 * j * LATTICE_PARAM))

    return Box(particles=particles, cube = cube)



def create_runner() -> RmcRunner:
    
    particle_factory = CylindricalParticleFactory(
        command_factory=CylindricalCommandFactory(nominal_step_size=RADIUS),
        cylinder_radius=RADIUS,
        cylinder_radius_polydispersity=RADIUS_POLYD,
        cylinder_height=HEIGHT,
        cylinder_height_polydispersity=0.0,
        cylinder_sld=0.0,
        solvent_sld=SOLVENT_SLD
    )
    box = create_box(PARTICLE_NUMBER, particle_factory=particle_factory)
    box_list = [box]
    data_src = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\20220204113917_p6mm_normalized_and_merged.txt"
    data = np.genfromtxt(data_src, delimiter='\t' )
    q = data[:,0]
    intensity = data[:,1]
    intensity_err = data[:,2]
    r_array = np.linspace(0, HEIGHT, num = 100)
    single_profile_calculator = ProfileCalculator(q_array=q, r_array=r_array)
    profile_fitter = ProfileFitter(box_list=box_list, single_profile_calculator=single_profile_calculator, experimental_intensity=intensity, intensity_uncertainty=intensity_err)
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
    return RmcRunner(
        logger = Logger(callback_list=[]),
        simulator=simulator,
        force_log=True)



if __name__ == "__main__":
    runner = create_runner()
    runner.run()
    simulation = runner.simulator.evaluator.simulation
    box_list = simulation.fitter.box_list
    box = box_list[0]
    plt.plot([p.position.x for p in box.particles],[p.position.y for p in box.particles], 'b.')
    plt.show()

#%%
