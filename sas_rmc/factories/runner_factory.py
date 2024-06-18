#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from datetime import datetime

import pandas as pd
import yaml

from sas_rmc.controller import Controller
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticle
from sas_rmc.scattering_simulation import ScatteringSimulation
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
from typing import Iterator, Any

import pandas as pd

from sas_rmc import Vector, constants
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.particles import CoreShellParticle
from sas_rmc.factories import parse_data
from sas_rmc.simulator import Simulator


rng = constants.RNG


def polydisperse_parameter(loc: float, polyd: float) -> float:
    return rng.normal(loc = loc, scale = loc * polyd)



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
    nominal_concentration: float
    particle_number: int
    box_number: int
    total_cycles: int
    annealing_type: str
    anneal_start_temp: float
    anneal_fall_rate: float
    annealing_stop_cycle_number: int
    detector_smearing: bool
    field_direction: str
    force_log_file: bool

    def create_particle(self) -> CoreShellParticle:
        return CoreShellParticle.gen_from_parameters(
            position=Vector.null_vector(),
            magnetization=self.core_magnetization * Vector.random_vector(),
            core_radius=polydisperse_parameter(self.core_radius, polyd=self.core_polydispersity),
            thickness=polydisperse_parameter(self.shell_thickness, polyd=self.shell_polydispersity),
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )
    
    def create_simulation_state(self) -> ScatteringSimulation:
        return ScatteringSimulation(
            scale_factor=self.nominal_concentration
        )
    
    def create_runner(self) -> RmcRunner:
        return RmcRunner(
            simulator=Simulator(
                controller=Controller(
                    commands=[],
                    acceptance_scheme=[]
                )

            )
        )
    
    @classmethod
    def create_from_dict(cls, d: dict):
        d_validated = {
            k : cls.__dataclass_fields__[k].type(v)
            for k, v in d.items() 
            if k in cls.__dataclass_fields__
        }
        if 'annealing_stop_cycle_number' not in d_validated:
            d_validated['annealing_stop_cycle_number'] = int(d_validated['total_cycles'] / 2)
        return cls(**d_validated)

def parse_value_frame(value_frame: pd.DataFrame) -> Iterator[tuple[str, Any]]:
    for _, row in value_frame.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        if not param_name.strip():
            continue
        if '#' in param_name.strip():
            continue
        v = param_value.strip()
        if not v:
            continue
        if v.lower() in parse_data.truth_dict:
            v = parse_data.truth_dict[v.lower()]
        yield param_name.strip(), v



def create_runner(input_config_path: Path) -> RmcRunner:

    print(f"Loading configuration from {input_config_path}, please wait a moment...")
    dataframes = pd.read_excel(
        input_config_path,
        dtype = str,
        sheet_name = None,
        keep_default_na=False,
        )
    dataframe = list(dataframes.values())[0]
    d = {k : v for k, v in parse_value_frame(dataframe)}
    return CoreShellRunner.create_from_dict(d)


if __name__ == "__main__":
    r = create_runner(r"E:\Programming\SasRMC\data\CoreShell_F20_pol.xlsx")

    assert type(False) == type(bool(False))
