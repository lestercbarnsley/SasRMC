#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import yaml
import pandas as pd

from .simulator import Simulator, timeit
#from .box_simulation import Box
#from .detector import DetectorImage
from .logger import LogCallback, Logger
from .simulator_factory import gen_config_from_dataframes, generate_file_path_maker
from .template_generator import generate_core_shell, generate_dumbbell, generate_reload


'''@dataclass
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
            logger.watch_simulation(self.simulator)'''


@dataclass
class Runner(ABC):

    @abstractmethod
    def run(self) -> None:
        pass


@dataclass
class RmcRunner(Runner):
    logger: Logger
    #callback_list: List[LogCallback]
    simulator: Simulator
    force_log: bool = True

    '''def generate_logger(self) -> Logger:
        return Logger(callback_list=self.callback_list)'''

    def run_force_log(self) -> None:
        with self.logger:#generate_logger() as l:
            #why doesnb't the context manager work?'
            #print(len(l.callback_list))
            self.simulator.simulate()

    def run_not_forced_log(self) -> None:
        #logger = Logger(self.callback_list)
        self.logger.before_event()
        self.simulator.simulate()
        self.logger.after_event()

    def run(self) -> None:
        if self.force_log:
            self.run_force_log()
        else:
            self.run_not_forced_log()


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




@timeit
def rmc_runner_factory(input_config_source: Path, output_path: Path) -> RmcRunner:
    print(f"Loading configuration from {input_config_source}, please wait a moment...")
    dataframes = pd.read_excel(
           input_config_source,
           dtype = str,
           sheet_name = None,
           keep_default_na=False,
        )
    simulator_config = gen_config_from_dataframes(dataframes)
    
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

TEMPLATE_PARAMS = {
    "generate core shell template":{
        "template_name" : "CoreShell_Simulation_Input",
        "template_generating_method" : generate_core_shell
        },
    "generate dumbbell template":{
        "template_name" : "Dumbell_Simulation_Input",
        "template_generating_method" : generate_dumbbell
        },
    "generate reload template":{
        "template_name" : "Reload_Simulation_Input",
        "template_generating_method" : generate_reload
        }
    }

def load_config(config_file: str) -> Runner:
    config_file_path = Path(config_file)
    with open(config_file_path, "r") as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)
    input_config_source = configs['input_config_source']
    if input_config_source in TEMPLATE_PARAMS:
        template_params = TEMPLATE_PARAMS[input_config_source]
        return TemplateGenerator(
            config_folder=config_file_path.parent,
            template_name=template_params["template_name"],
            template_generating_method=template_params["template_generating_method"])
    input_config_path = Path(input_config_source)
    if not input_config_path.exists():
        input_config_path = config_file_path.parent / input_config_path
    output_path = configs["output_folder"]
    if r'./' in output_path:
        output_path = config_file_path.parent / Path(output_path.replace(r'./', ''))
    return rmc_runner_factory(input_config_path, output_path)


if __name__ == "__main__":
    pass
