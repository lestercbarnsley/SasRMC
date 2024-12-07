#%%
from typing import Iterator
import random

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc.controller import ControlStep, Controller
from sas_rmc.factories import parse_data
from sas_rmc.factories.command_factory import CommandFactory
from sas_rmc.factories.acceptable_command_factory import AcceptanceFactory
from sas_rmc.scattering_simulation import ScatteringSimulation


def particle_box_index_iterator(simulation_state: ScatteringSimulation) -> Iterator[tuple[int, int]]:
    for box_index, box in enumerate(simulation_state.box_list):
        for particle_index, _ in enumerate(box.particle_results):
            yield box_index, particle_index


@pydantic_dataclass
class ControllerFactory:
    command_factory: CommandFactory
    acceptance_factory: AcceptanceFactory
    total_cycles: int

    def create_control_steps(self, simulation_state: ScatteringSimulation) -> Iterator[ControlStep]:
        for cycle in range(self.total_cycles):
            particle_box_indices = list(particle_box_index_iterator(simulation_state))
            for step, (box_index, particle_index) in enumerate(random.sample(particle_box_indices, len(particle_box_indices))):
                command = self.command_factory.create_command(simulation_state, box_index, particle_index)
                acceptance_scheme = self.acceptance_factory.create_acceptance(cycle, step)
                yield ControlStep(command, acceptance_scheme)

    def create_controller(self, simulation_state: ScatteringSimulation) -> Controller:
        return Controller(
            ledger=[step for step in self.create_control_steps(simulation_state)]
        )
    
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        command_factory = CommandFactory.create_from_dataframes(dataframes)
        acceptance_factory = AcceptanceFactory.create_from_dataframes(dataframes)
        return cls(
            command_factory = command_factory,
            acceptance_factory = acceptance_factory,
            **value_frame)
    

