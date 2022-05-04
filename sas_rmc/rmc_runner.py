from pathlib import Path
import yaml

import pandas as pd

from .logger import Logger
from .simulator_factory import SimulationConfig

class RmcRunner: # Not a dataclass
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
        self.force_log = simulator_config.force_log

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


def load_config(config_file: str) -> RmcRunner:
    return RmcRunner(config_file)

if __name__ == "__main__":
    pass
