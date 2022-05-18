#%%

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Protocol

import yaml
import pandas as pd

from .simulator import Simulator
from .box_simulation import Box
from .detector import DetectorImage
from .logger import Logger
from .simulator_factory import SimulationConfig, generate_file_path_maker
from .template_generator import generate_core_shell, generate_dumbbell, generate_numerical_dumbbell, generate_reload


@dataclass
class RmcRunner:
    detector_list: List[DetectorImage]
    box_list: List[Box]
    save_file_maker: Callable[[str, str], Path]
    simulator: Simulator
    force_log: bool = True
    output_format: str = None

    def generate_logger(self) -> Logger:
        return Logger(
            box_list=self.box_list,
            controller=self.simulator.controller,
            save_path_maker=self.save_file_maker,
            detector_list=self.detector_list,
            output_format=self.output_format
        )

    def run(self) -> None:
        if self.force_log:      
            with self.generate_logger():
                self.simulator.simulate()
        else:
            logger = self.generate_logger()
            logger.watch_simulation(self.simulator)

@dataclass
class TemplateGenerator:
    config_folder: Path
    template_name: str
    template_generating_method: Callable[[Path], None]

    def run(self) -> None:
        file_path_maker = generate_file_path_maker(self.config_folder, description=self.template_name)
        file = file_path_maker("", "xlsx")
        self.template_generating_method(file)
        print(f"{self.template_name} file generated at {file}.")
        print(f"Find your config.yaml and change the 'input_config_source' field to {file.name}")
        print(f"Edit and save {file.name} to configure your simulation.")
        print(f"If you're having difficulty reading parameters in Excel, select top row and try Home -> Format -> Autofit Column Width")


@dataclass
class Runner(Protocol):
    def run(self) -> None:
        pass


def rmc_runner_factory(input_config_source: Path, output_path: Path) -> RmcRunner:
    print(f"Loading configuration from {input_config_source}, please wait a moment...")
    dataframes = pd.read_excel(
           input_config_source,
           dtype = str,
           sheet_name = None,
           keep_default_na=False,
        )
    simulator_config = SimulationConfig.gen_from_dataframes(dataframes)
    
    detector_list = simulator_config.generate_detector_list(dataframes)
    box_list = simulator_config.generate_box_list(detector_list)
    save_file_maker = simulator_config.generate_save_file_maker(output_path)
    scattering_sim = simulator_config.generate_scattering_simulation(detector_list, box_list)
    controller = simulator_config.generate_controller(scattering_sim, box_list)
    simulator = simulator_config.generate_simulator(
        controller=controller,
        simulation=scattering_sim,
        box_list=box_list
    )
    force_log = simulator_config.force_log
    output_format = None if simulator_config.output_plot_format == "none" else simulator_config.output_plot_format
    print("Configuration loaded. Simulation starting.")
    return RmcRunner(
        detector_list=detector_list,
        box_list=box_list,
        save_file_maker=save_file_maker,
        simulator=simulator,
        force_log=force_log,
        output_format=output_format
    )



def core_shell_factory(data_folder) -> TemplateGenerator:
    template_name = "CoreShell_Simulation_Input"
    return TemplateGenerator(
        config_folder=data_folder,
        template_name=template_name,
        template_generating_method=generate_core_shell
    )

def dumbbell_factory(data_folder) -> TemplateGenerator:
    template_name = "Dumbbell_Simulation_Input"
    return TemplateGenerator(
        config_folder=data_folder,
        template_name=template_name,
        template_generating_method=generate_dumbbell
    )

def load_config(config_file: str) -> Runner:
    config_file_path = Path(config_file)
    with open(config_file_path, "r") as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)
    if configs['input_config_source'] == "generate core shell template":
        return core_shell_factory(config_file_path.parent)
    if configs['input_config_source'] == "generate dumbbell template":
        return dumbbell_factory(config_file_path.parent)
    if configs['input_config_source'] == "generate numerical dumbbell template":
        return generate_numerical_dumbbell(config_file_path.parent)
    if configs['input_config_source'] == "generate reload template":
        return generate_reload(config_file_path.parent)
    input_config_source = Path(configs['input_config_source'])
    if not input_config_source.exists():
        input_config_source = config_file_path.parent / input_config_source
    output_path = configs["output_folder"]
    if r'./' in output_path:
        output_path = config_file_path.parent / Path(output_path.replace(r'./', ''))
    return rmc_runner_factory(input_config_source, output_path)


if __name__ == "__main__":
    pass
