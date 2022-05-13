from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Protocol

import yaml
import pandas as pd

from .template_generator import df_list_to_excel, generate_core_shell_template
from .simulator import Simulator
from .box_simulation import Box
from .detector import DetectorImage
from .logger import Logger
from .simulator_factory import SimulationConfig, generate_save_file

@dataclass
class RmcRunner:
    detector_list: List[DetectorImage]
    box_list: List[Box]
    save_file_path: Path
    simulator: Simulator
    force_log: bool = True
    '''
    def __init__(self, config_file: str) -> None:
        config_file_path = Path(config_file)
        with open(config_file_path, "r") as f:
            configs = yaml.load(f, Loader = yaml.FullLoader)
        input_config_source = Path(configs['input_config_source'])
        if not input_config_source.exists():
            input_config_source = config_file_path.parent / input_config_source
        dataframes = pd.read_excel(
           input_config_source,
           dtype = str,
           sheet_name = None,
           keep_default_na=False,
        )
        simulator_config = SimulationConfig.gen_from_dataframes(dataframes)
        
        self.detector_list = simulator_config.generate_detector_list(dataframes)
        self.box_list = simulator_config.generate_box_list(self.detector_list)
        output_path = configs["output_folder"]
        if r'./' in output_path:
            output_path = config_file_path.parent / Path(output_path.replace(r'./', ''))
        self.save_file_path = simulator_config.generate_save_file(output_path)
        scattering_sim = simulator_config.generate_scattering_simulation(self.detector_list, self.box_list)
        controller = simulator_config.generate_controller(scattering_sim, self.box_list)
        self.simulator = simulator_config.generate_simulator(
            controller=controller,
            simulation=scattering_sim,
            box_list=self.box_list
        )
        self.force_log = simulator_config.force_log'''

    def run(self) -> None:
        if self.force_log:      
            with Logger(
                box_list = self.box_list,
                controller=self.simulator.controller,
                save_file_path=self.save_file_path ,
                detector_list=self.detector_list
            ):
                self.simulator.simulate()
        else:
            logger = Logger(
                box_list = self.box_list,
                controller=self.simulator.controller,
                save_file_path=self.save_file_path ,
                detector_list=self.detector_list
            )
            logger.watch_simulation(self.simulator)

@dataclass
class TemplateGenerator:
    config_folder: Path
    template_name: str
    template_generating_method: Callable[[Path], None]

    def run(self) -> None:
        file = generate_save_file(self.config_folder, self.template_name, file_format="xlsx")
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
    save_file_path = simulator_config.generate_save_file(output_path)
    scattering_sim = simulator_config.generate_scattering_simulation(detector_list, box_list)
    controller = simulator_config.generate_controller(scattering_sim, box_list)
    simulator = simulator_config.generate_simulator(
        controller=controller,
        simulation=scattering_sim,
        box_list=box_list
    )
    force_log = simulator_config.force_log
    print("Configuration loaded. Simulation starting.")
    return RmcRunner(
        detector_list=detector_list,
        box_list=box_list,
        save_file_path=save_file_path,
        simulator=simulator,
        force_log=force_log
    )

def generate_core_shell(output_path: Path) -> None:
    dfs, sheet_names = generate_core_shell_template()
    df_list_to_excel(output_path, dfs, sheet_names)

def core_shell_factory(data_folder) -> TemplateGenerator:
    template_name = "CoreShell_Simulation_Input"
    return TemplateGenerator(
        config_folder=data_folder,
        template_name=template_name,
        template_generating_method=generate_core_shell
    )

def load_config(config_file: str) -> Runner:
    config_file_path = Path(config_file)
    with open(config_file_path, "r") as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)
    if configs['input_config_source'] == "generate core shell template":
        return core_shell_factory(config_file_path.parent)
    input_config_source = Path(configs['input_config_source'])
    if not input_config_source.exists():
        input_config_source = config_file_path.parent / input_config_source
    output_path = configs["output_folder"]
    if r'./' in output_path:
        output_path = config_file_path.parent / Path(output_path.replace(r'./', ''))
    return rmc_runner_factory(input_config_source, output_path)

if __name__ == "__main__":
    pass
