#%%
from dataclasses import dataclass
from typing import List

from .acceptable_command_factory import AcceptableCommandFactory
from .. import commands, constants
from ..controller import Controller
from ..scattering_simulation import SimulationParams
from ..box_simulation import Box
from . import annealing_config# import AnnealingConfig, gen_from_dict
from .particle_factory import ParticleFactory

rng = constants.RNG


@dataclass
class ControllerFactory:
    annealing_config: annealing_config.AnnealingConfig
    total_cycles: int
    p_factory: ParticleFactory
    acceptable_command_factory: AcceptableCommandFactory

    def create_controller(self, simulation_params: SimulationParams, box_list: List[Box]) -> Controller:
        ledger = [
            commands.SetSimulationParams(simulation_params, change_to_factors = simulation_params.values),
            ]
        for box in box_list:
            for particle_index, _ in enumerate(box.particles):
                ledger.append(
                    commands.SetParticleState.gen_from_particle(box, particle_index)
                )
        
        for cycle in range(self.total_cycles):
            box_ledger = []
            temperature = self.annealing_config.get_temperature(cycle)
            for box_index, box in enumerate(box_list):
                for particle_index, _ in enumerate(box.particles):
                    command = self.p_factory.create_command(box, particle_index, simulation_params)
                    acceptable_command = self.acceptable_command_factory.create_acceptable_command(command, temperature)
                    acceptable_command.update_loggable_data(
                        {"Cycle": cycle, "Box index": box_index}
                    )
                    box_ledger.append(acceptable_command)
            rng.shuffle(box_ledger)
            ledger += box_ledger    

        return Controller(
            ledger=ledger
        )


def gen_from_dict(d: dict, p_factory: ParticleFactory, acceptable_command_factory: AcceptableCommandFactory) -> ControllerFactory:
    annealing = annealing_config.gen_from_dict(d)
    total_cycles = d.get("total_cycles")
    return ControllerFactory(annealing, total_cycles=total_cycles, p_factory=p_factory, acceptable_command_factory=acceptable_command_factory)
    

