#%%

from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Iterable, Iterator, ParamSpec, TypeVar
import random

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass
import numpy as np

from sas_rmc import constants, Evaluator, commands, Controller, ControlStep
from sas_rmc.loggers import logger
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticleForm
from sas_rmc.factories import parse_data, box_factory, particle_factory, command_factory, acceptable_command_factory, evaluator_factory
from sas_rmc.simulator import Simulator


rng = constants.RNG
DATETIME_FORMAT = '%Y%m%d%H%M%S'


def polydisperse_parameter(loc: float, polyd: float) -> float:
    return rng.normal(loc = loc, scale = loc * polyd)


P = ParamSpec('P')
R = TypeVar('R')

def create_command_if_acceptable_command(command_factory: Callable[P, R], command_types: Iterable[type]) -> Callable[P, R]:

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
    commands.RescaleCommand,
    commands.RelativeRescale,
    #commands.RescaleBoxMagnetization
}

create_core_shell_command = create_command_if_acceptable_command(command_factory.create_command, core_shell_commands)

def particle_box_index_iterator(simulation_state: ScatteringSimulation) -> Iterator[tuple[int, int]]:
    for box_index, box in enumerate(simulation_state.box_list):
        for particle_index, _ in enumerate(box.particles):
            yield box_index, particle_index


@pydantic_dataclass
class CoreShellRunner:
    simulation_title: str
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

    def create_particle(self) -> CoreShellParticleForm:
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
    
    def create_simulation_state(self, default_box_dimensions: list[float] | None = None) -> ScatteringSimulation:
        box_dimensions = [self.box_dimension_1, self.box_dimension_2, self.box_dimension_3]
        if np.prod(box_dimensions) == 0:
            box_dimensions = default_box_dimensions
        if box_dimensions is None:
            raise ValueError("Box dimensions are missing.")
        return ScatteringSimulation(
            scale_factor=SimulationParam( 1.0, name = "scale_factor", bounds = (0, np.inf)),
            box_list=box_factory.create_box_list(self.create_particle, box_dimensions, self.particle_number, self.box_number, self.nominal_concentration )
        )
    
    def create_control_steps(self, simulation_state: ScatteringSimulation) -> Iterator[ControlStep]:
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number > 0 else int(self.total_cycles / 2)
        temperature = self.anneal_start_temp
        if self.annealing_type.lower() == "greedy".lower():
            temperature = 0
        for cycle in range(self.total_cycles):
            if cycle > annealing_stop_cycle:
                temperature = 0
            particle_box_indices = list(particle_box_index_iterator(simulation_state))
            for step, (box_index, particle_index) in enumerate(random.sample(particle_box_indices, len(particle_box_indices))):
                box = simulation_state.box_list[box_index]
                command = create_core_shell_command(
                    box_index = box_index,
                    particle_index=particle_index,
                    move_by_distance=self.core_radius,
                    cube = box.cube,
                    total_particle_number=len(box.particles),
                    nominal_magnetization=self.core_magnetization
                )
                acceptance_scheme = acceptable_command_factory.create_metropolis_acceptance(temperature, cycle, step)
                yield ControlStep(command, acceptance_scheme)
            temperature = temperature * (1- self.anneal_fall_rate)
            if "very".lower() not in self.annealing_type.lower():
                temperature = self.anneal_start_temp / (1 + cycle)
    
    def create_runner(self, evaluator: Evaluator, results_folder: Path) -> RmcRunner:
        state = self.create_simulation_state(default_box_dimensions=evaluator.default_box_dimensions())
        datetime_string = datetime.now().strftime(DATETIME_FORMAT)
        log_callback = logger.LogEventBus(
            log_callbacks=[
                logger.QuietLogCallback(), 
                logger.ExcelCallback(excel_file= results_folder / Path(f'{datetime_string}_{self.simulation_title}.xlsx')),
                logger.BoxPlotter(result_folder=results_folder, file_plot_prefix=f'{datetime_string}_{self.simulation_title}'),
                logger.DetectorImagePlotter(result_folder=results_folder, file_plot_prefix=f'{datetime_string}_{self.simulation_title}'),
                logger.ProfilePlotter(result_folder=results_folder, file_plot_prefix=f'{datetime_string}_{self.simulation_title}')
                ]
        )
        return RmcRunner(
            simulator=Simulator(
                controller=Controller(ledger=[step for step in self.create_control_steps(state)]),
                state = state,
                evaluator=evaluator,
                log_callback=log_callback
                )

            )
    
    @classmethod
    def create_from_dataframe(cls, dataframe: pd.DataFrame):
        d = parse_data.parse_value_frame(dataframe)
        return cls(**d)


def create_runner(input_config_path: Path, result_folder: Path) -> RmcRunner:

    dataframes = pd.read_excel(
        input_config_path,
        dtype = str,
        sheet_name = None,
        keep_default_na=False,
        )
    value_frame = list(dataframes.values())[0]
    runner_factory = CoreShellRunner.create_from_dataframe(value_frame)
    evaluator = (evaluator_factory.create_evaluator_with_smearing(dataframes) 
                 if runner_factory.detector_smearing 
                 else evaluator_factory.create_evaluator_no_smearing(dataframes))
    return runner_factory.create_runner(evaluator, result_folder)
    


if __name__ == "__main__":
    #data_params = create_runner(r"E:\Programming\SasRMC\data\CoreShell_F20_pol.xlsx")
    spreadsheet = Path(__file__).parent.parent.parent / Path("data") / Path("CoreShell Simulation Input - Copy - Copy.xlsx")
    #spreadsheet = Path(__file__).parent.parent.parent / Path("data") / Path("CoreShell_F20_pol - Copy.xlsx")
    #runner = create_runner(spreadsheet)
    #runner.run()

    '''import cProfile
    import pstats

    with cProfile.Profile() as pr:
        runner.run()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    '''
    

# %%
