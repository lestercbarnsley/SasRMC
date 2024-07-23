#%%
from pathlib import Path
from typing import Callable
from datetime import datetime
import random

import numpy as np
import pandas as pd
import yaml
from pydantic.dataclasses import dataclass

from sas_rmc import acceptance_scheme, commands, constants
from sas_rmc.controller import Controller
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticle
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam


rng = constants.RNG
'''
from .acceptable_command_factory import MetropolisAcceptanceFactory
from . import box_factory, particle_factory, particle_factory_spherical, simulation_factory, controller_factory, simulator_factory, detector_builder, parse_data
from ..rmc_runner import Runner, RmcRunner
from ..logger import BoxPlotter, Logger, ExcelCallback, DetectorPlotter, ProfilePlotter


@dataclass
class RunnerFactory(ABC):
    
    @abstractmethod
    def create_runner(self, input_config_source: Path, output_path: Path) -> Runner:
        pass



def generate_save_file(datetime_string: str, output_folder: Path, description: str = "", comment: str = "", file_format: str = "xlsx") -> Path:
    return output_folder / Path(f"{datetime_string}_{description}{comment}.{file_format}")

def generate_file_path_maker(output_folder: Path, description: str = "") -> Callable[[str, str], Path]:
    datetime_format = '%Y%m%d%H%M%S'
    datetime_string = datetime.now().strftime(datetime_format)
    return lambda comment, file_format: generate_save_file(datetime_string, output_folder, description, comment, file_format)




PARTICLE_TYPE_DICT = {
    "CoreShellParticle" : particle_factory_spherical.CoreShellParticleFactory
}

def dict_to_particle_factory(d: dict) -> particle_factory.ParticleFactory:
    particle_type = PARTICLE_TYPE_DICT[d.get("particle_type")]
    return particle_type.gen_from_dict(d)

RESULT_MAKER_DICT = {
    "Analytical" : lambda particle_factory : detector_builder.AnalyticalResultMakerFactory(),
    "Numerical" : lambda particle_factory : detector_builder.NumericalResultMakerFactory(particle_factory)
}

def dict_to_result_maker_factory(d: dict, particle_factory: particle_factory.ParticleFactory) -> detector_builder.ResultMakerFactory:
    return RESULT_MAKER_DICT[d.get("calculator_type","Analytical")](particle_factory)





@dataclass
class RMCRunnerFactory(RunnerFactory):

    def create_runner(self, input_config_source: Path, output_path: Path) -> RmcRunner:
        
        print(f"Loading configuration from {input_config_source}, please wait a moment...")
        dataframes = pd.read_excel(
            input_config_source,
            dtype = str,
            sheet_name = None,
            keep_default_na=False,
            )
        dataframe = list(dataframes.values())[0]
        config_dict = parse_data.dataframe_to_config_dict(dataframe)
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


def load_config(config_file: str) -> Runner:
    config_file_path = Path(config_file)
    with open(config_file_path, "r") as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)
    input_config_source = configs['input_config_source']
    
    input_config_path = Path(input_config_source)
    if not input_config_path.exists():
        input_config_path = config_file_path.parent / input_config_path
    output_path = configs["output_folder"]
    if r'./' in output_path:
        output_path = config_file_path.parent / Path(output_path.replace(r'./', ''))
    runner_factory = RMCRunnerFactory()
    return runner_factory.create_runner(input_config_source = input_config_path, output_path=output_path)
'''
from pathlib import Path
from typing import Iterator, Any, ParamSpec, TypeVar

import pandas as pd

from sas_rmc import Vector, constants, evaluator
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.particles import CoreShellParticle
from sas_rmc.factories import parse_data, box_factory, particle_factory, command_factory, acceptable_command_factory
from sas_rmc.simulator import Simulator


rng = constants.RNG


def polydisperse_parameter(loc: float, polyd: float) -> float:
    return rng.normal(loc = loc, scale = loc * polyd)


P = ParamSpec('P')
R = TypeVar('R')

def create_command_if_acceptable_command(command_factory: Callable[P, R], command_types: list[type]) -> Callable[P, R]:

    def new_command_factory(*args: P.args, **kwargs: P.kwargs) -> R:
        for _ in range(2_000_000):
            command = command_factory(*args, **kwargs)
            if type(command) in command_types:
                return command
        raise TypeError(f"Factory function {command_factory.__name__} is incompatible with command_types")
    return new_command_factory


core_shell_commands = {
    commands.MoveParticleBy,
    commands.MoveParticleTo,
    commands.JumpParticleTo,
    commands.OrbitParticle,
    commands.MagnetizeParticle,
    commands.RescaleMagnetization,
    commands.RotateMagnetization,
    commands.RescaleCommand
}

create_core_shell_command = create_command_if_acceptable_command(command_factory.create_command, core_shell_commands)

def particle_box_index_iterator(simulation_state: ScatteringSimulation) -> Iterator[tuple[int, int]]:
    for box_index, box in enumerate(simulation_state.box_list):
        for particle_index, _ in enumerate(box.particles):
            yield box_index, particle_index


@dataclass
class CoreShellRunner:
    core_radius: float
    core_polydispersity: float
    core_sld: float
    shell_thickness: float
    shell_polydispersity: float
    shell_sld: float
    solvent_sld: float
    core_magnetization: float
    total_cycles: int
    annealing_type: str
    anneal_start_temp: float
    anneal_fall_rate: float
    detector_smearing: bool
    field_direction: str
    force_log_file: bool
    annealing_stop_cycle_number: int = -1
    nominal_concentration: float = 0.0
    particle_number: int = 0
    box_number: int = 0
    box_dimension_1: float = 0.0
    box_dimension_2: float = 0.0
    box_dimension_3: float = 0.0

    def create_particle(self) -> CoreShellParticle:
        return particle_factory.create_core_shell_particle(
            core_radius=self.core_radius,
            core_polydispersity=self.core_polydispersity,
            core_sld=self.core_sld,
            shell_thickness=self.shell_thickness,
            shell_polydispersity=self.shell_polydispersity,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld,
            core_magnetization=self.core_magnetization
        )
    
    def create_simulation_state(self) -> ScatteringSimulation:
        return ScatteringSimulation(
            scale_factor=SimulationParam(self.nominal_concentration if self.nominal_concentration else 1.0, name = "scale_factor", bounds = (0, np.inf)),
            box_list=box_factory.create_box_list(self.create_particle, [self.box_dimension_1, self.box_dimension_2, self.box_dimension_3], self.particle_number, self.box_number, self.nominal_concentration )
        )

    def create_commands(self, simulation_state: ScatteringSimulation) -> Iterator[commands.Command]:
        for _ in range(self.total_cycles):
            particle_box_indices = list(particle_box_index_iterator(simulation_state))
            for particle_index, box_index in random.sample(particle_box_indices, len(particle_box_indices)):
                box = simulation_state.box_list[box_index]
                yield create_core_shell_command(
                    box_index = box_index,
                    particle_index=particle_index,
                    move_by_distance=self.core_radius,
                    cube = box.cube,
                    total_particle_number=len(box.particles),
                    nominal_magnetization=self.core_magnetization
                )

    def create_acceptance_scheme(self, simulation_state: ScatteringSimulation) -> Iterator[acceptance_scheme.AcceptanceScheme]:
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number < 0 else int(self.total_cycles / 2)
        temperature = self.anneal_start_temp
        if self.annealing_type.lower() == "greedy".lower():
            temperature = 0
        for cycle in range(self.total_cycles):
            if cycle > annealing_stop_cycle:
                temperature = 0
            for step, _ in enumerate(particle_box_index_iterator(simulation_state)):
                yield acceptable_command_factory.create_metropolis_acceptance(temperature, cycle, step)
            temperature = temperature * (1- self.anneal_fall_rate)
            if "very".lower() not in self.annealing_type.lower():
                temperature = self.anneal_start_temp / (1 + cycle)
            

    
    def create_runner(self, evaluator: evaluator.Evaluator) -> RmcRunner:
        state = self.create_simulation_state()
        return RmcRunner(
            simulator=Simulator(
                controller=Controller(
                    commands=[c for c in self.create_commands(state)],
                    acceptance_scheme=[a for a in self.create_acceptance_scheme(state)]
                ),
                state = state,
                evaluator=evaluator,
                )

            )
    
    @classmethod
    def create_from_dataframe(cls, dataframe: pd.DataFrame):
        d = parse_data.parse_value_frame(dataframe)
        return cls(**d)


def create_runner(input_config_path: Path) -> RmcRunner:

    print(f"Loading configuration from {input_config_path}, please wait a moment...")
    dataframes = pd.read_excel(
        input_config_path,
        dtype = str,
        sheet_name = None,
        keep_default_na=False,
        )
    value_frame = list(dataframes.values())[0]
    runner_factory = CoreShellRunner.create_from_dataframe(value_frame)
    
    return runner_factory.create_runner()


if __name__ == "__main__":
    from sas_rmc.factories.detector_builder import create_detector_image
    #data_params = create_runner(r"E:\Programming\SasRMC\data\CoreShell_F20_pol.xlsx")
    spreadsheet = Path(__file__).parent.parent.parent / Path("data") / Path("CoreShell_F20_pol.xlsx")
    dataframes = pd.read_excel(
        spreadsheet,
        dtype = str,
        sheet_name = None,
        keep_default_na=False,
        )
    
    explore = list(dataframes.values())[2]

    from sas_rmc.detector import DetectorImage
    #create_runner(spreadsheet)
    df = dataframes['Data parameters']
    d = create_detector_image(dataframes['M3-polDown-20m'],{k : v for k, v in dataframes['Data parameters'].iloc[0].items()})

    #d = DetectorImage.gen_from_pandas(dataframes['M3-polDown-20m'])


    



# %%
