#%%

from pathlib import Path
from collections.abc import Callable
from typing import Iterable, Iterator, ParamSpec, TypeVar
import random

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass
import numpy as np

from sas_rmc import constants, Evaluator, commands, acceptance_scheme, Controller, logger
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.particles import CoreShellParticle
from sas_rmc.factories import parse_data, box_factory, particle_factory, command_factory, acceptable_command_factory, evaluator_factory
from sas_rmc.simulator import Simulator


rng = constants.RNG


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

    def create_commands(self, simulation_state: ScatteringSimulation) -> Iterator[commands.Command]:
        for _ in range(self.total_cycles):
            particle_box_indices = list(particle_box_index_iterator(simulation_state))
            for box_index, particle_index in random.sample(particle_box_indices, len(particle_box_indices)):
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
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number > 0 else int(self.total_cycles / 2)
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
            

    
    def create_runner(self, evaluator: Evaluator) -> RmcRunner:
        state = self.create_simulation_state(default_box_dimensions=evaluator.default_box_dimensions())
        return RmcRunner(
            simulator=Simulator(
                controller=Controller(
                    commands=[c for c in self.create_commands(state)],
                    acceptance_scheme=[a for a in self.create_acceptance_scheme(state)]
                ),
                state = state,
                evaluator=evaluator,
                log_callback=logger.QuietLogCallback()
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
    evaluator = (evaluator_factory.create_evaluator_with_smearing(dataframes) 
                 if runner_factory.detector_smearing 
                 else evaluator_factory.create_evaluator_no_smearing(dataframes))
    return runner_factory.create_runner(evaluator)
    


if __name__ == "__main__":
    #data_params = create_runner(r"E:\Programming\SasRMC\data\CoreShell_F20_pol.xlsx")
    #spreadsheet = Path(__file__).parent.parent.parent / Path("data") / Path("CoreShell Simulation Input - Copy - Copy.xlsx")
    spreadsheet = Path(__file__).parent.parent.parent / Path("data") / Path("CoreShell_F20_pol - Copy.xlsx")
    runner = create_runner(spreadsheet)
    runner.run()

    



# %%
