

from dataclasses import dataclass, field
from typing import Callable, List, Protocol
from pathlib import Path

import pandas as pd

from .simulator import Simulator
from .controller import Controller, CommandProtocol
from .detector import DetectorImage
from .box_simulation import Box
from .command_writer import BoxWriter, CommandWriter, Loggable


class LoggableCommandProtocol(CommandProtocol, Loggable, Protocol):
    pass # combine the commandprotocol with loggable, only in this context


def _log_command(command_writer: CommandWriter, command: LoggableCommandProtocol) -> dict:
    command.execute()
    return command_writer.to_data(command)


@dataclass
class Logger:
    box_list: List[Box]
    controller: Controller
    save_path_maker: Callable[[str, str],Path]
    detector_list: List[DetectorImage]
    output_format: str = None
    command_writer: CommandWriter = field(default_factory=CommandWriter.standard_particle_writer)
    box_writer: BoxWriter = field(default_factory=BoxWriter.standard_box_writer)
    _particle_states_log: List[pd.DataFrame] = field(default_factory=list, init=False, repr=False)
    
    def log(self) -> pd.DataFrame:
        data = [_log_command(self.command_writer, command) for command in self.controller.completed_commands]
        return pd.DataFrame(data)

    @property
    def particle_states(self) -> List[pd.DataFrame]:
        return [pd.DataFrame(self.box_writer.to_data(box)) for box in self.box_list]

    def save_plots(self):
        for box_number, box in enumerate(self.box_list):
            fig = self.box_writer.to_plot(box)
            fig_path = self.save_path_maker(f"_box_{box_number}_particle_positions", self.output_format)
            fig.savefig(fig_path)
        for detector_number, detector in enumerate(self.detector_list):
            detector_fig = detector.plot_intensity(show_fig=False)
            detector_fig_path = self.save_path_maker(f"_detector_{detector_number}", self.output_format)
            detector_fig.savefig(detector_fig_path)

    def __enter__(self):
        self._particle_states_log = self.particle_states
        return self

    def write_output(self) -> None:
        if self.output_format:
            self.save_plots()
        with pd.ExcelWriter(self.save_path_maker("", "xlsx")) as writer:
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
        self._particle_states_log = self.particle_states
        simulator.simulate()
        self.write_output()

if __name__ == "__main__":
    pass

