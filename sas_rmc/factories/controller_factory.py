#%%
from abc import ABC, abstractmethod
from typing import Iterator
import random

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc.controller import ControlStep, Controller
from sas_rmc.factories import acceptable_command_factory, parse_data
from sas_rmc.factories.command_factory import CommandFactory
from sas_rmc.scattering_simulation import ScatteringSimulation


@pydantic_dataclass
class ControllerFactory(ABC):

    @abstractmethod
    def create_controller(self, simulation_state: ScatteringSimulation) -> Controller:
        pass


def particle_box_index_iterator(simulation_state: ScatteringSimulation) -> Iterator[tuple[int, int]]:
    for box_index, box in enumerate(simulation_state.box_list):
        for particle_index, _ in enumerate(box.particle_results):
            yield box_index, particle_index


@pydantic_dataclass
class ControllerFactoryConcrete(ControllerFactory):
    command_factory: CommandFactory
    annealing_stop_cycle_number: int
    anneal_start_temp: float
    annealing_type: str
    total_cycles: int
    anneal_fall_rate: float

    def create_greedy_temperature(self, cycle: int) -> float:
        return 0
    
    def create_very_fast_temperature(self, cycle: int) -> float:
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number > 0 else int(self.total_cycles / 2)
        if cycle > annealing_stop_cycle:
            return 0
        return self.anneal_start_temp * ((1- self.anneal_fall_rate)**cycle)
    
    def create_fast_temperature(self, cycle: int) -> float:
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number > 0 else int(self.total_cycles / 2)
        if cycle > annealing_stop_cycle:
            return 0
        return self.anneal_start_temp / (1 + cycle)
    
    def get_temperature(self, cycle: int) -> float:
        if self.annealing_type.lower() == "greedy".lower():
            return self.create_greedy_temperature(cycle)
        if "very".lower() not in self.annealing_type.lower():
            return self.create_very_fast_temperature(cycle)
        return self.create_fast_temperature(cycle)

    def create_control_steps(self, simulation_state: ScatteringSimulation) -> Iterator[ControlStep]:
        for cycle in range(self.total_cycles):
            temperature = self.get_temperature(cycle)
            particle_box_indices = list(particle_box_index_iterator(simulation_state))
            for step, (box_index, particle_index) in enumerate(random.sample(particle_box_indices, len(particle_box_indices))):
                command = self.command_factory.create_command(simulation_state, box_index, particle_index)
                acceptance_scheme = acceptable_command_factory.create_metropolis_acceptance(temperature, cycle, step)
                yield ControlStep(command, acceptance_scheme)

    def create_controller(self, simulation_state: ScatteringSimulation) -> Controller:
        return Controller(
            ledger=[step for step in self.create_control_steps(simulation_state)]
        )
    
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        command_factory = CommandFactory.create_from_dataframes(dataframes)
        return cls(command_factory = command_factory, **value_frame)
    

