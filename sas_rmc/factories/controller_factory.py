#%%
from abc import ABC, abstractmethod

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc.controller import Controller
from sas_rmc.scattering_simulation import ScatteringSimulation


@pydantic_dataclass
class ControllerFactory(ABC):
    @abstractmethod
    def create_controller(self, simulation_state: ScatteringSimulation) -> Controller:
        pass

    @classmethod
    @abstractmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        pass


@pydantic_dataclass
class ControllerFactoryConcrete(ControllerFactory):
    annealing_stop_cycle_number: int
    anneal_start_temp: float
    annealing_type: str
    total_cycles: int

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
                    total_particle_number=len(box.particle_results),
                    nominal_magnetization=self.core_magnetization
                )
                acceptance_scheme = acceptable_command_factory.create_metropolis_acceptance(temperature, cycle, step)
                yield ControlStep(command, acceptance_scheme)
            temperature = temperature * (1- self.anneal_fall_rate)
            if "very".lower() not in self.annealing_type.lower():
                temperature = self.anneal_start_temp / (1 + cycle)