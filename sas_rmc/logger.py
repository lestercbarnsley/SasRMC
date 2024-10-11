#%%

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sas_rmc.vector import Vector


@dataclass
class LogCallback:

    @abstractmethod
    def start(self, document: dict  | None = None) -> None:
        pass

    @abstractmethod
    def event(self, document: dict  | None = None) -> None:
        pass

    @abstractmethod
    def stop(self, document: dict  | None = None) -> None:
        pass


@dataclass
class NoLogCallback(LogCallback):
    def start(self, document: dict | None = None) -> None:
        pass

    def event(self, document: dict | None = None) -> None:
        pass

    def stop(self, document: dict | None = None) -> None:
        pass

@dataclass
class PrintLogCallback(LogCallback):

    def start(self, document: dict | None = None) -> None:
        print('start', document)

    def event(self, document: dict | None = None) -> None:
        print('event', document)
        
    def stop(self, document: dict | None = None) -> None:
        print('event', document)


        


@dataclass
class QuietLogCallback(LogCallback):

    def start(self, document: dict | None = None) -> None:
        pass
        #print('start', document)

    def event(self, document: dict | None = None) -> None:
        if document is None:
            return None
        print({k : v for k, v in document.items() if k in ['Current goodness of fit', 'Cycle', 'Step', 'timestamp']})

    def stop(self, document: dict | None = None) -> None:
        pass


@dataclass
class LogEventBus(LogCallback):
    log_callbacks: list[LogCallback]

    def start(self, document: dict  | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.start(document)

    def event(self, document: dict | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.event(document)

    def stop(self, document: dict | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.stop(document)


def particle_data_to_magnetization_vector(particle_data: dict) -> Vector | None:
    try:
        return Vector(
            x = particle_data['Magnetization.X'],
            y = particle_data['Magnetization.Y'],
            z = particle_data['Magnetization.Z']
        )
    except KeyError:
        return None


@dataclass
class BoxData:
    particle_list: list
    dim_list: list

    def to_dataframe(self) -> pd.DataFrame:
        dim_data = {f'box_dimension_{i}' : dim for i, dim in enumerate(self.dim_list)}
        particle_data_list = [particle_data | (dim_data if i == 0 else {}) for i, particle_data in enumerate(self.particle_list)]
        return pd.DataFrame(data = particle_data_list)
    
    def get_all_magnetization(self) -> list[Vector]:
        all_vectors = [particle_data_to_magnetization_vector(particle_data) for particle_data in self.particle_list]
        return [vector for vector in all_vectors if vector is not None]
    
    def get_all_volume(self) -> list[float]:
        return [particle['Volume'] for particle in self.particle_list]

    def calculate_concentration(self) -> float:
        total_particle_volume = np.sum(self.get_all_volume())
        box_volume = np.prod(self.dim_list)
        return float(total_particle_volume / box_volume)

    @classmethod
    def create_from_dict(cls, d):
        particle_list = []
        dim_list = []
        for key in d:
            if 'particle' in key.lower():
                particle_list.append(d[key])
            elif 'dim' in key.lower():
                dim_list.append(d[key])
        return BoxData(
            particle_list=particle_list,
            dim_list=dim_list
        )
    

@dataclass
class SimData:
    box_data_list: list[BoxData]
    scale_factor_data: dict

    def to_dataframes(self) -> list[pd.DataFrame]:
        return [box_data.to_dataframe() for box_data in self.box_data_list]
    
    def get_scale_factor_value(self) -> float:
        return self.scale_factor_data['Value']

    def calculate_corrected_concentration(self) -> float:
        return float(
            self.get_scale_factor_value() * np.average(
                [box_data.calculate_concentration() for box_data in self.box_data_list]
                )
            )
    
    def get_average_magnetization(self) -> Vector:
        vectors = sum((box_data.get_all_magnetization() for box_data in self.box_data_list), start= [])
        total_particles = len(vectors)
        vector_sum = sum(vectors, start = Vector.null_vector())
        return vector_sum / total_particles

    @classmethod
    def create_from_dict(cls, d):
        simulation_data = d.get('scattering_simulation', {})
        box_data_list = []
        scale_factor_data = d.get('scale_factor', {})
        for key in simulation_data:
            if 'box_' in key.lower():
                box_data_list.append(BoxData.create_from_dict(simulation_data[key]))
        return SimData(box_data_list=box_data_list, scale_factor_data=scale_factor_data)


def event_docs_to_simulation_data(event_docs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(data = event_docs)

def fitter_data_to_detector_data(data: dict) -> list[dict]:
    detector_data = data['Detector data']
    simulated_intensity = data['Simulated intensity']
    polarization = data['Polarization']
    return [
        det_data | {'simulated_intensity' : sim_intensity} | ({'Polarization' : polarization} if i == 0 else {})
        for i, (det_data, sim_intensity) in enumerate(zip(detector_data, simulated_intensity))
        ]

def start_stop_docs_to_detector_image_data(docs: list[dict]) -> list[pd.DataFrame]:
    doc = docs[-1]
    fitter_doc = doc['Fitter']
    return [
        pd.DataFrame(fitter_data_to_detector_data(fitter_doc[fitter_key])) 
        for fitter_key in sorted(fitter_doc.keys()) 
        if 'fitter' in fitter_key.lower()
    ]

def sim_data_to_global_df(sim_data: SimData, total_time: float) ->  pd.DataFrame:
    
    final_rescale = sim_data.get_scale_factor_value()
    estimated_concentration = sim_data.calculate_corrected_concentration()

    average_magnetization = sim_data.get_average_magnetization()
    
    d = {
        "Final scale": [final_rescale],
        "Estimated concentration (v/v)": [estimated_concentration],
        "Simulation time (s)": [total_time],
        "AverageMagnetization.X" : [average_magnetization.x],
        "AverageMagnetization.Y" : [average_magnetization.y],
        "AverageMagnetization.Z" : [average_magnetization.z]
    }
    return pd.DataFrame(d)


@dataclass
class ExcelCallback(LogCallback):
    excel_file: Path
    start_docs: list[dict] = field(default_factory=list)
    event_docs: list[dict] = field(default_factory=list)
    stop_docs: list[dict] = field(default_factory=list)

    def start(self, document: dict | None = None) -> None:
        if document:
            self.start_docs.append(document)

    def event(self, document: dict | None = None) -> None:
        if document:
            self.event_docs.append(document)

    def stop(self, document: dict | None = None) -> None:
        if document:
            self.stop_docs.append(document)
        with pd.ExcelWriter(self.excel_file) as writer:
            start_sim_data = SimData.create_from_dict(self.start_docs[-1])
            for box_number, initial_particle_state_df in enumerate(start_sim_data.to_dataframes()):
                initial_particle_state_df.to_excel(writer, sheet_name=f"Box {box_number} Initial Particle States")
            simulation_df = event_docs_to_simulation_data(self.event_docs)
            simulation_df.to_excel(writer, sheet_name="Simulation Data")
            final_sim_data = SimData.create_from_dict(self.stop_docs[-1])
            for box_number, final_particle_state_df in enumerate(final_sim_data.to_dataframes()):
                final_particle_state_df.to_excel(writer, sheet_name=f"Box {box_number} Final Particle States")
            detector_image_data_dfs = start_stop_docs_to_detector_image_data(self.stop_docs)
            for detector_number, detector_data_df in enumerate(detector_image_data_dfs):
                detector_data_df.to_excel(writer, sheet_name= f"Final detector image {detector_number}")
            total_time = max(event_doc.get('timestamp', 0) for event_doc in self.event_docs) - min(event_doc.get('timestamp', np.inf) for event_doc in self.event_docs)
            
            global_df = sim_data_to_global_df(final_sim_data, total_time = total_time)
            global_df.to_excel(writer, sheet_name="Global parameters Final")


@dataclass
class BoxPlotter(LogCallback):
    result_folder: Path

    def start(self, document: dict | None = None) -> None:
        pass

    def event(self, document: dict | None = None) -> None:
        pass

    def stop(self, document: dict | None = None) -> None:
        return super().stop(document)


'''
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

def make_global_params(box_list: List[Box], before_time: datetime, commands: List[AcceptableCommand]) -> pd.DataFrame:
    final_rescale = 1
    test_command = lambda acc : isinstance(acc, AcceptableCommand) and acc.acceptance_scheme.is_acceptable() and isinstance(acc.base_command, ScaleCommand)
    test_commands = [acceptable_command for acceptable_command in commands if test_command(acceptable_command)]
    for acceptable_command in test_commands:
        final_rescale = acceptable_command.base_command.simulation_params.get_value(key = constants.NUCLEAR_RESCALE)
    estimated_concentration = final_rescale * np.average([np.sum([particle.volume for particle in box.particles ]) / box.volume for box in box_list])
    d = {
        "Final scale": [final_rescale],
        "Estimated concentration (v/v)": [estimated_concentration],
        "Simulation time (s)": [(datetime.now() - before_time).total_seconds()],
        "Effective magnetization" : ["Coming soon"]
    }
    return pd.DataFrame(d)




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
    make_initial: bool = False

    def plot_box_list(self, timing_string: str = "initial") -> None:
        box_writer = BoxWriter.standard_box_writer()
        for box_number, box in enumerate(self.box_list):
            fig = box_writer.to_plot(box)
            fig_path = self.save_path_maker(f"_box_{box_number}_{timing_string}_particle_positions", self.format)
            fig.savefig(fig_path)

    def before_event(self, d: dict = None) -> None:
        if self.make_initial:
            self.plot_box_list(timing_string="initial")

    def after_event(self, d: dict = None) -> None:
        self.plot_box_list(timing_string="final")


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


'''

if __name__ == "__main__":
    print(datetime.now().timestamp())

