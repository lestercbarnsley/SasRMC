#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import pandas as pd

from .acceptable_command_factory import MetropolisAcceptanceFactory
from . import box_factory, particle_factory, spherical_particle_factory, simulation_factory, controller_factory, simulator_factory
from ..rmc_runner import Runner, RmcRunner


@dataclass
class RunnerFactory(ABC):
    
    @abstractmethod
    def create_runner(input_config_source: Path, output_path: Path) -> Runner:
        pass


  
truth_dict = {
    'ON' : True,
    'OFF' : False,
    'on' : True,
    'off' : False,
    'On': True,
    'Off': False,
    'True' :  True,
    'TRUE' : True,
    'true' : True,
    'False' :  False,
    'FALSE' : False,
    'false' : False
} # I'm sure I haven't come close to fully covering all the wild and creative ways users could say "True" or "False"

def is_bool_in_truth_dict(s: str) -> bool:
    """Checks if a string can be converted to a bool

    Looks up a string against a set of interpretable options and decides if the string can be converted to a bool

    Parameters
    ----------
    s : str
        A string to test

    Returns
    -------
    bool
        Returns True if the string can be interpreted as a bool
    """
    return s in truth_dict     



def is_int(s: str) -> bool:
    """Checks if a string is compatible with the int format

    Parameters
    ----------
    s : str
        A string that might be converted to a int

    Returns
    -------
    bool
        True if the string can be converted to a int
    """
    return _is_numeric_type(s, t = int)

def add_row_to_dict(d: dict, param_name: str, param_value: str) -> None:
    if not param_name.strip():
        return
    if r'#' in param_name.strip():
        return
    v = param_value.strip()
    if not v:
        return
    if is_int(v):
        d[param_name] = int(v)
    elif is_float(v):
        d[param_name] = float(v)
    elif is_bool_in_truth_dict(v):
        d[param_name] = truth_dict[v]
    else:
        d[param_name] = v

def dataframe_to_config_dict(dataframe: pd.DataFrame) -> dict:
    config_dict = dict()
    for _, row in dataframe.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        add_row_to_dict(config_dict, param_name, param_value)
    return config_dict

PARTICLE_TYPE_DICT = {
    "CoreShellParticle" : spherical_particle_factory.CoreShellParticleFactory
}


def dict_to_particle_factory(d: dict) -> particle_factory.ParticleFactory:
    particle_type = PARTICLE_TYPE_DICT[d.get("particle_type")]
    return particle_type.gen_from_dict(d)


@dataclass
class RMCRunnerFactory(RunnerFactory):

    def create_runner(input_config_source: Path, output_path: Path) -> Runner:
        print(f"Loading configuration from {input_config_source}, please wait a moment...")
        dataframes = pd.read_excel(
            input_config_source,
            dtype = str,
            sheet_name = None,
            keep_default_na=False,
            )
        dataframe, dataframe_2 = list(dataframes.values())[:2]
        config_dict = dataframe_to_config_dict(dataframe)
        particle_factory = dict_to_particle_factory(config_dict)
        box_factory_ = box_factory.gen_from_dict(config_dict)
        box_list_factory = box_factory.gen_list_factory_from_dict(config_dict)
        simulation_factory_ = simulation_factory.gen_from_dict(config_dict, )
        controller_factory_ = controller_factory.gen_from_dict(config_dict, particle_factory, acceptable_command_factory=MetropolisAcceptanceFactory())
        simulator_factory_ = simulator_factory.MemorizedSimulatorFactory()
        box_list = box_list_factory.create_box_list(box_factory_, particle_factory)

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


def rmc_runner_factory(input_config_source: Path, output_path: Path) -> RmcRunner: