#%%
# from dataclasses import dataclass
# from typing import List

from typing import Callable, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sas_rmc
from sas_rmc import constants, Vector
from sas_rmc.factories.controller_factory import ControllerFactory
from sas_rmc.factories.acceptable_command_factory import MetropolisAcceptanceFactory
from sas_rmc.factories.annealing_config import VeryFastAnneal
from sas_rmc.factories.particle_factory_cylindrical import  CylindricalCommandFactory, ParticleFactory, CylindricalLongParticleFactory
from sas_rmc.factories.simulator_factory import MemorizedSimulatorFactory
from sas_rmc.factories.runner_factory import generate_file_path_maker 
from sas_rmc.box_simulation import Box, Cube
from sas_rmc.profile_calculator import ProfileFitter
from sas_rmc.result_calculator import ProfileCalculatorAnalytical
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.controller import Controller
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.logger import Logger, LogCallback , controller_writer, box_writer

rng = constants.RNG
PI = constants.PI


@dataclass
class ProfileCallback(LogCallback):
    save_path_maker: Callable[[str, str],Path]
    box_list: List[Box]
    simulation: ScatteringSimulation
    controller: Controller
    before_event_log: List[Tuple[str,pd.DataFrame]] = field(default_factory = list, repr = False, init = False)
    before_time: datetime = field(default_factory= datetime.now, repr= False, init= False)

    def before_event(self, d: dict = None) -> None:
        sheet_name_writer = lambda box_number : f"Box {box_number} Initial Particle States"
        self.before_event_log.extend([(sheet_name_writer(box_number), box_writer(box)) for box_number, box in enumerate(self.box_list)])
        self.before_time = datetime.now()

    def get_scattering_data(self) -> dict:
        fitter = self.simulation.fitter
        q = fitter.single_profile_calculator.q_array
        intensity = fitter.experimental_intensity
        uncertainty = fitter.experimental_uncertainty()
        simulated_intensity = fitter.simulated_intensity(self.simulation.simulation_params.get_value(constants.NUCLEAR_RESCALE))
        return {
            "Q" : q,
            "Intensity" : intensity,
            "Uncertainty" : uncertainty,
            "Simulated intensity" : simulated_intensity
        }

    def after_event(self, d: dict = None) -> None:
        excel_file = self.save_path_maker("", "xlsx")
        with pd.ExcelWriter(excel_file) as writer:
            for sheet_name, before_event_log_i in self.before_event_log:
                before_event_log_i.to_excel(writer, sheet_name=sheet_name)
            controller_log = controller_writer(self.controller)
            controller_log.to_excel(writer, sheet_name = "Simulation data")
            for box_number, box in enumerate(self.box_list):
                box_log = box_writer(box)
                box_log.to_excel(writer, sheet_name = f"Box {box_number} Final Particle States")
            detector_data = self.get_scattering_data()
            pd.DataFrame(detector_data).to_excel(writer, sheet_name = "Scattering data")
        fig, ax = plt.subplots()
        ax.loglog(detector_data["Q"], detector_data["Intensity"], 'b.')
        ax.loglog(detector_data["Q"], detector_data["Simulated intensity"], 'r-')
        fig.set_size_inches(6,4)
        ax.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  16)#'x-large')
        ax.set_ylabel(r'Intensity (cm $^{-1}$)',fontsize =  16)#'x-large')
        fig.tight_layout()
        plt.show()
        fig.savefig(self.save_path_maker("scattering intensity", "pdf"))
        for box_number, box in enumerate(self.box_list):
            fig, ax = plt.subplots()
            ax.scatter(x = [particle.position.x for particle in box.particles], y = [particle.position.y for particle in box.particles])
            fig.set_size_inches(5,5)
            ax.set_xlabel(r'X ($\AA$)',fontsize =  16)#'x-large')
            ax.set_ylabel(r'Y ($\AA$)',fontsize =  16)#'x-large')
            fig.tight_layout()
            fig.savefig(self.save_path_maker(f"particle_positions_box_{box_number}", "pdf"))
            plt.show()
        


    
        


RADIUS = 100/2#68.0 /2 # ANGSTROM
RADIUS_POLYD = 0.1# 10%
LATTICE_PARAM = 4 * PI / (np.sqrt(3) *  0.0707 )  #100#    / (np.sqrt(3) / 2)#92.0
print(LATTICE_PARAM)
HEIGHT = 0.4e4# is a micron#100e3#300000.0 ### Make this extremely long?
SOLVENT_SLD = 8.29179504046
PARTICLE_NUMBER =100#0
CYCLES = 50#0 
RUN = True
STARTING_RESCALE = 0.1

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

def create_random_box(particle_number: int, particle_factory: ParticleFactory) -> Box:
    particles = [particle_factory.create_particle() for _ in range(particle_number)]
    cube = Cube(dimension_0=TWOPIDIVQMIN, dimension_1=TWOPIDIVQMIN, dimension_2=HEIGHT)
    box = Box(particles=particles, cube = cube)
    for i, _ in enumerate(box.particles):
        while box.wall_or_particle_collision(i):
            particle = box.particles[i]
            random_position = box.cube.random_position_inside()
            new_position = Vector(random_position.x, random_position.y)
            box.particles[i] = particle.set_position(new_position)
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
    box_maker = lambda : create_box(PARTICLE_NUMBER, particle_factory=particle_factory)
    box_maker = lambda : create_hexagonal_box(PARTICLE_NUMBER, particle_factory)
    #box_maker = lambda : create_random_box(PARTICLE_NUMBER, particle_factory)
    box_list = [box_maker() for _ in range(3)]
    print('starting config collision found',[box.collision_test() for box in box_list])
    data_src = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\20220204113917_p6mm_normalized_and_merged.txt"
    data = np.genfromtxt(data_src, delimiter='\t' )
    nominal_q_min = 2 * PI / np.min([[box.cube.dimension_0, box.cube.dimension_1] for box in box_list])
    nominal_q_min = 0.02
    q = data[:,0]
    q_limited = lambda arr : np.array([a for a, q_i in zip(arr, q) if q_i > nominal_q_min])
    #q_limited = lambda arr: arr
    intensity = data[:,1]
    intensity_err = data[:,2]
    single_profile_calculator = ProfileCalculatorAnalytical(q_limited(q))
    profile_fitter = ProfileFitter(box_list=box_list, single_profile_calculator=single_profile_calculator, experimental_intensity=q_limited(intensity), intensity_uncertainty=q_limited(intensity_err))
    simulation = ScatteringSimulation(profile_fitter, simulation_params=sas_rmc.box_simulation_params_factory(STARTING_RESCALE))
    controller_factory = ControllerFactory(
        annealing_config=VeryFastAnneal(annealing_stop_cycle_number=CYCLES / 2,anneal_start_temp=200.0, anneal_fall_rate=0.1),
        total_cycles=CYCLES,
        p_factory=particle_factory,
        acceptable_command_factory=MetropolisAcceptanceFactory()
        )
    controller = controller_factory.create_controller(simulation.simulation_params, box_list)
    simulator_factory = MemorizedSimulatorFactory(controller, simulation, box_list)
    simulator = simulator_factory.create_simulator()
    profile_callback = ProfileCallback(
        save_path_maker=generate_file_path_maker(output_folder=r'.\data\results'),
        box_list = box_list, 
        simulation = simulation, 
        controller = controller
        )
    return RmcRunner(
        logger = Logger(callback_list=[profile_callback]),
        simulator=simulator,
        force_log=True)





if __name__ == "__main__":
    runner = create_runner()
    if RUN:
        runner.run()
    else:
        runner.logger.before_event()
        runner.logger.after_event()


#%%
