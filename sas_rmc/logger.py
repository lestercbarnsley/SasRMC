

from dataclasses import dataclass, field
from typing import List, Protocol
from pathlib import Path

import pandas as pd

from .simulator import Simulator
from .controller import Controller, CommandProtocol
from .detector import DetectorImage
from .box_simulation import Box
from .particle import Particle
from .command_writer import BoxWriter, CommandWriter, Loggable


def default_particle_dictionary(particle: Particle, box_number: int = -1):
    return {
        'Box index': box_number,
        'Particle type': type(particle).__name__,
        'Position.X': particle.position.x,
        'Position.Y': particle.position.y,
        'Position.Z': particle.position.z,
        'Orientation.X': particle.orientation.x,
        'Orientation.Y':particle.orientation.y,
        'Orientation.Z': particle.orientation.z,
        'Magnetization.X':particle._magnetization.x,
        'Magnetization.Y':particle._magnetization.y,
        'Magnetization.Z':particle._magnetization.z,
        'Volume':particle.volume,
        'Total scattering length': particle.scattering_length
    }

def box_to_pandas(box: Box, box_number: int = -1) -> pd.DataFrame:
    return pd.DataFrame(
        [default_particle_dictionary(p, box_number=box_number) for p in box.particles]
    )

def box_list_to_panda_list(box_list: List[Box]) -> List[pd.DataFrame]:
    return [box_to_pandas(box, box_number) for box_number, box in enumerate(box_list)]


class LoggableCommandProtocol(CommandProtocol, Loggable, Protocol):
    pass # combine the commandprotocol with loggable, only in this context

def _log_command(command_writer: CommandWriter, command: LoggableCommandProtocol) -> dict:
    command.execute()
    return command_writer.to_data(command)


@dataclass
class Logger:
    box_list: List[Box]
    controller: Controller
    save_file_path: Path
    detector_list: List[DetectorImage]
    command_writer: CommandWriter = field(default_factory=CommandWriter.standard_particle_writer)
    box_writer: BoxWriter = field(default_factory=BoxWriter.standard_box_writer)
    _particle_states_log: List[pd.DataFrame] = field(default_factory=list, init=False, repr=False)

    def log(self) -> pd.DataFrame:
        data = [_log_command(self.command_writer, command) for command in self.controller.completed_commands]
        return pd.DataFrame(data)

    @property
    def particle_states(self) -> List[pd.DataFrame]:
        return [pd.DataFrame(self.box_writer.to_data(box)) for box in self.box_list]

    def __enter__(self):
        self._particle_states_log = self.particle_states
        return self

    def write_output(self) -> None:
        with pd.ExcelWriter(self.save_file_path) as writer:
            for i, initial_particle_log in enumerate(self._particle_states_log):
                initial_particle_log.to_excel(writer, sheet_name=f"Box {i} Initial Particle States")
            self.log().to_excel(writer, sheet_name="Simulation Data")
            for i, final_particle_log in enumerate(self.particle_states):
                final_particle_log.to_excel(writer, sheet_name=f"Box {i} Final Particle States")
            for i, d in enumerate(self.detector_list):
                d.get_pandas().to_excel(writer, sheet_name=f"Final detector image {i}")

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.write_output() # Will skip writing log if exception raised and force log is False

    def watch_simulation(self, simulator: Simulator) -> None:
        self._particle_states_log = box_list_to_panda_list(self.box_list)
        simulator.simulate()
        self.write_output()

if __name__ == "__main__":
    pass

