

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from .simulator import Simulator
from .controller import Controller, CommandProtocol
from .detector import DetectorImage, SimulatedDetectorImage
from .box_simulation import Box
from .command_writer import BoxWriter, CommandWriter, Loggable
from . import constants

PI = constants.PI


class LoggableCommandProtocol(CommandProtocol, Loggable, Protocol):
    pass # combine the commandprotocol with loggable, only in this context


def _log_command(command_writer: CommandWriter, command: LoggableCommandProtocol) -> dict:
    command.execute()
    return command_writer.to_data(command)


@dataclass
class LogCallback:

    @abstractmethod
    def before_event(self, d: dict = None) -> None:
        pass

    @abstractmethod
    def after_event(self, d: dict = None) -> None:
        pass


def box_writer(box: Box) -> pd.DataFrame:
    box_df = pd.DataFrame([p.get_loggable_data() for p in box.particles])
    return box_df

def detector_writer(detector: DetectorImage) -> pd.DataFrame:
    detector_df = detector.get_pandas()
    return detector_df

def get_loggable_commmand(command: LoggableCommandProtocol) -> dict:
    command.execute()
    return command.get_loggable_data()

def controller_writer(controller: Controller) -> pd.DataFrame:
    return pd.DataFrame([get_loggable_commmand(command) for command in controller.completed_commands])


@dataclass
class ExcelCallback(LogCallback):
    save_path_maker: Callable[[str, str],Path]
    box_list: List[Box]
    detector_list: List[DetectorImage]
    controller: Controller
    before_event_log: List[Tuple[str,pd.DataFrame]] = field(default_factory = list, repr = False, init = False)

    def before_event(self, d: dict = None) -> None:
        sheet_name_writer = lambda box_number : f"Box {box_number} Initial Particle States"
        self.before_event_log.extend([(sheet_name_writer(box_number), box_writer(box)) for box_number, box in enumerate(self.box_list)])

    def after_event(self, d: dict = None) -> None:
        excel_file = self.save_path_maker("", "xlsx")
        with pd.ExcelWriter(excel_file) as writer:
            for sheet_name, before_event_log_i in self.before_event_log:
                before_event_log_i.to_excel(writer, sheet_name=sheet_name)
            controller_log = controller_writer(self.controller)
            controller_log.to_excel(writer, sheet_name = "Simulation data")
            for box_number, box in enumerate(self.box_list):
                box_log = box_writer(box)
                box_log.to_excel(writer, sheet_name = f"Box {box_number} Final Particle States")
            for detector_number, detector in enumerate(self.detector_list):
                detector_log = detector_writer(detector)
                detector_log.to_excel(writer, sheet_name= f"Final detector image {detector_number}")


def plot_box(box: Box, file_name: Path) -> None:
    box_writer = BoxWriter.standard_box_writer()
    fig = box_writer.to_plot(box)
    fig.savefig(file_name)


@dataclass
class BoxPlotter(LogCallback):
    save_path_maker: Callable[[str, str],Path]
    box_list: List[Box]
    format: str = "pdf"
    make_initial: bool = True

    def plot_boxes(self, comment_maker: Callable[[int], Path]) -> None:
        for box_number, box in enumerate(self.box_list):
            comment = comment_maker(box_number)
            plot_box(box, self.save_path_maker(comment, self.format))

    def before_event(self, d: dict = None) -> None:
        if not self.make_initial:
            return
        comment_maker = lambda box_number : f"_box_{box_number}_initial_particle_positions"
        self.plot_boxes(comment_maker)

    def after_event(self, d: dict = None) -> None:
        comment_maker = lambda box_number : f"_box_{box_number}_final_particle_positions"
        self.plot_boxes(comment_maker)


def plot_detector(detector: DetectorImage, file_name: Path) -> None:
    fig = detector.plot_intensity(show_fig=False)
    fig.savefig(file_name)


@dataclass
class DetectorPlotter(LogCallback):
    save_path_maker: Callable[[str, str],Path]
    detector_list: List[DetectorImage]
    format: str = "pdf"
    make_initial: bool = False

    def plot_detectors(self, comment_maker: Callable[[int], Path]) -> None:
        for detector_number, detector in enumerate(self.detector_list):
            comment = comment_maker(detector_number)
            plot_detector(detector, self.save_path_maker(comment, self.format))

    def before_event(self, d: dict = None) -> None:
        if not self.make_initial:
            return
        comment_maker = lambda detector_number : f"_detector_{detector_number}_initial"
        self.plot_detectors(comment_maker)

    def after_event(self, d: dict = None) -> None:
        comment_maker = lambda detector_number : f"_detector_{detector_number}_final"
        self.plot_detectors(comment_maker)

def interpolate_qxqy(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, shadow: np.ndarray, qx_target: float, qy_target: float) -> Optional[float]:
    distances = np.sqrt((qx - qx_target)**2 + (qy - qy_target)**2)
    arg_of_min = np.where(distances == np.amin(distances))
    return intensity[arg_of_min] if shadow[arg_of_min] else None


def sector_average(detector: DetectorImage, intensity_2d: Callable[[DetectorImage], Tuple[np.ndarray, np.ndarray, np.ndarray]] = None, sector_tuple: Tuple[float, float] = (0, np.inf), num_per_rotation: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    sector_tuple = sector_tuple if sector_tuple[1] < 2 * PI else (0, 2 *PI)
    nominal_sector_angle, sector_range = sector_tuple   
    qx, qy, intensity, shadow = intensity_2d(detector) if intensity_2d is not None else lambda detector : detector.intensity_2d()
    detector_q_arr = np.sqrt((qx[shadow])**2 + (qy[shadow])**2)
    q_list = np.arange(start = np.min(detector_q_arr), stop = np.max(detector_q_arr), step = np.min(detector.qxqy_delta))
    def intensity_at_q(q: float) -> float:
        angle_num = int(num_per_rotation * sector_range / (2* PI))
        angle_min = nominal_sector_angle - sector_range / 2
        angle_max = nominal_sector_angle + sector_range / 2
        intensities = []
        angles = np.linspace(angle_min, angle_max, num = angle_num)
        for angle in np.concatenate([angles, angles + PI]):
            qx_target, qy_target = q * np.cos(angle), q * np.sin(angle)
            if np.min(qx) < qx_target < np.max(qx) and np.min(qy) < qy_target < np.max(qy):
                intensity_out = interpolate_qxqy(qx, qy, intensity, shadow, qx_target, qy_target)
                if intensity_out is not None:
                    intensities.append(intensity_out)
        return np.average(intensities) if len(intensities) else 0.0
    tensity_list_fn = np.frompyfunc(intensity_at_q, nin = 1, nout = 1)
    tensity_list = tensity_list_fn(q_list)
    tensity_filter = lambda arr: arr[tensity_list!=0]
    return tensity_filter(q_list), tensity_filter(tensity_list)

def plot_profile(detector: SimulatedDetectorImage, file_name: Path, angle_list: List[float] = None) -> None:
    fig, ax = plt.subplots()
    angle_list = angle_list if angle_list is not None else [i * PI/6 for i in range(4)]
    colours = [c for c in mcolors.BASE_COLORS.keys() if c!='w'] * 50
    factors = [i * 2 for i, _ in enumerate(angle_list)]
    for factor, colour, angle in zip(factors, colours, angle_list):
        q, exp_int = sector_average(detector, lambda d: d.intensity_2d(), sector_tuple=[angle, PI/10])
        q_sim, sim_int = sector_average(detector, lambda d: d.simulated_intensity_2d(), sector_tuple=[angle, PI/10])
        ax.loglog(q, (10**factor) * exp_int, colour + '.' , label = f"{(180/PI)*angle:.2f} deg")
        ax.loglog(q_sim, (10**factor) * sim_int, colour + '-')
    fig.set_size_inches(5,5)
    ax.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  16)#'x-large')
    ax.set_ylabel(r'Intensity (cm $^{-1}$)',fontsize =  16)#'x-large')
    ax.legend()
    fig.savefig(file_name)
    

@dataclass
class ProfilePlotter(LogCallback): # This is bad code, a better solution is composition, but the point is, I have to write the bad code before I can write the good code
    save_path_maker: Callable[[str, str],Path]
    detector_list: List[DetectorImage]
    format: str = "pdf"
    make_initial: bool = False

    def plot_detectors(self, comment_maker: Callable[[int], Path]) -> None:
        for detector_number, detector in enumerate(self.detector_list):
            comment = comment_maker(detector_number)
            plot_profile(detector, self.save_path_maker(comment, self.format))

    def before_event(self, d: dict = None) -> None:
        if not self.make_initial:
            return
        comment_maker = lambda detector_number : f"_profiles_{detector_number}_initial"
        self.plot_detectors(comment_maker)

    def after_event(self, d: dict = None) -> None:
        comment_maker = lambda detector_number : f"_profiles_{detector_number}_final"
        self.plot_detectors(comment_maker)


@dataclass
class BoxPlotter(LogCallback):
    save_path_maker: Callable[[str, str],Path]
    box_list: List[Box]
    format: str = "pdf"
    make_initial: bool = True

    def plot_box_list(self) -> None:
        box_writer = BoxWriter.standard_box_writer()
        for box_number, box in enumerate(self.box_list):
            fig = box_writer.to_plot(box)
            fig_path = self.save_path_maker(f"_box_{box_number}_particle_positions", self.format)
            fig.savefig(fig_path)

    def before_event(self, d: dict = None) -> None:
        pass

    def after_event(self, d: dict = None) -> None:
        self.plot_box_list()


@dataclass
class Logger:

    callback_list: List[LogCallback] = field(default_factory=list)

    def add_callback(self, log_callback: LogCallback):
        self.callback_list.append(log_callback)

    def before_event(self) -> None:
        for bec in self.callback_list:
            bec.before_event()

    def after_event(self) -> None:
        for bec in self.callback_list:
            bec.after_event()

    def __enter__(self):
        self.before_event()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.after_event()



@dataclass
class Logger_:
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

