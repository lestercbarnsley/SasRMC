#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from datetime import datetime

import pandas as pd
import yaml

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


if __name__ == "__main__":
    pass

