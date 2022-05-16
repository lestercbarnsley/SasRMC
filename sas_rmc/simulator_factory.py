
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Type, Union

import numpy as np
import pandas as pd

from .array_cache import array_cache
from . import commands
from .acceptance_scheme import MetropolisAcceptance
from .viewer import CLIViewer
from .scattering_simulation import MAGNETIC_RESCALE, NUCLEAR_RESCALE, Fitter2D, ScatteringSimulation, SimulationParam, SimulationParams
from .simulator import MemorizedSimulator, MonteCarloEvaluator, Simulator
from .vector import Vector
from .particle import CoreShellParticle, Particle
from .box_simulation import Box
from .detector import DetectorImage, Polarization, DetectorConfig, SimulatedDetectorImage
from .shapes import Cube
from .controller import Controller

rng = np.random.default_rng()
PI = np.pi
RANGE_FACTOR = 1.2

ParticleFactory = Callable[[], Particle]
BoxFactory = Callable[[int,ParticleFactory], Box]

DEFAULT_VAL_TYPES =  {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    Polarization: Polarization.UNPOLARIZED
} # Mark for deletion
  
truth_dict = {
    'ON' : True,
    'OFF' : False,
    'on' : True,
    'off' : False,
    'On': True,
    'Off': False,
    'True' :  True,
    'TRUE' : True,
    'true' : True,
    'False' :  False,
    'FALSE' : False,
    'false' : False
} # I'm sure I haven't come close to fully covering all the wild and creative ways users could say "True" or "False"

def is_bool_in_truth_dict(s: str) -> bool:
    """Checks if a string can be converted to a bool

    Looks up a string against a set of interpretable options and decides if the string can be converted to a bool

    Parameters
    ----------
    s : str
        A string to test

    Returns
    -------
    bool
        Returns True if the string can be interpreted as a bool
    """
    return s in truth_dict     

def _is_numeric_type(s: str, t: Type) -> bool:
    try:
        t(s)
        return True
    except ValueError:
        return False

def is_float(s: str) -> bool:
    """Checks if a string is compatible with the float format

    Parameters
    ----------
    s : str
        A string that might be converted to a float

    Returns
    -------
    bool
        True if the string can be converted to a float
    """
    return _is_numeric_type(s, t = float)

def is_int(s: str) -> bool:
    """Checks if a string is compatible with the int format

    Parameters
    ----------
    s : str
        A string that might be converted to a int

    Returns
    -------
    bool
        True if the string can be converted to a int
    """
    return _is_numeric_type(s, t = int)

def add_row_to_dict(d: dict, param_name: str, param_value: str) -> None:
    if not param_name.strip():
        return
    if r'#' in param_name.strip():
        return
    v = param_value.strip()
    if not v:
        return
    if is_int(v):
        d[param_name] = int(v)
    elif is_float(v):
        d[param_name] = float(v)
    elif is_bool_in_truth_dict(v):
        d[param_name] = truth_dict[v]
    else:
        d[param_name] = v

def dataframe_to_config_dict(dataframe: pd.DataFrame) -> dict:
    config_dict = dict()
    for _, row in dataframe.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        add_row_to_dict(config_dict, param_name, param_value)
    return config_dict

def different_random_int(n: int, number_to_avoid: int) -> int:
    for _ in range(200000):
        x = rng.choice(range(n))
        if x != number_to_avoid:
            return x
    return -1

def box_simulation_params_factory() -> SimulationParams:
    params = [
        SimulationParam(value = 1, name = NUCLEAR_RESCALE, bounds=(0, np.inf)), 
        SimulationParam(value = 1, name = MAGNETIC_RESCALE, bounds=(0, np.inf))
        ]
    return SimulationParams(params = params)

@array_cache
def qxqy_array_from_ranges(qx_min: float, qx_max: float, qx_delta: float, qy_min: float, qy_max: float, qy_delta: float, range_factor: float = RANGE_FACTOR, resolution_increase_factor: float = 1):
    qX, qY = np.meshgrid(
        np.arange(start = range_factor * qx_min, stop = range_factor * qx_max, step = qx_delta / resolution_increase_factor),
        np.arange(start = range_factor * qy_min, stop = range_factor * qy_max, step = qy_delta / resolution_increase_factor)
    )
    return qX, qY

def qxqy_array_from_detectorimage(detector_image: DetectorImage, range_factor: float = RANGE_FACTOR, resolution_increase_factor: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    qx_max = np.max(detector_image.qX)
    qx_min = np.min(detector_image.qX)
    delta_qx = detector_image.qx_delta
    qy_max = np.max(detector_image.qY)
    qy_min = np.min(detector_image.qY)
    delta_qy = detector_image.qy_delta
    qX, qY = qxqy_array_from_ranges(qx_min, qx_max, delta_qx, qy_min, qy_max, delta_qy, range_factor=range_factor, resolution_increase_factor=resolution_increase_factor)
    return qX, qY


def qxqy_array_list_from_detector_list(detector_list: List[DetectorImage], range_factor: float = RANGE_FACTOR, resolution_increase_factor: float = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [qxqy_array_from_detectorimage(detector_image, range_factor=range_factor, resolution_increase_factor=resolution_increase_factor) for detector_image in detector_list]

def command_factory(nominal_step_size: float, box: Box, particle_index: int, temperature: float, simulation_params: SimulationParams, spherical_particle: bool = False, magnetic_simulation: bool = False) -> commands.AcceptableCommand:
    nominal_angle_change = PI/8
    nominal_rescale= 0.02#0.005
    change_by_factor = rng.normal(loc = 1.0, scale = nominal_rescale)
    position_delta = Vector.random_vector_xy(rng.normal(loc = 0, scale = nominal_step_size) )
    move_by = lambda : commands.MoveParticleBy(
        box=box,
        particle_index=particle_index,
        position_delta=position_delta
    )
    jump_to = lambda : commands.JumpParticleTo(
        box = box,
        particle_index=particle_index,
        reference_particle_index=different_random_int(len(box), particle_index)
    )
    orbit = lambda : commands.OrbitParticle(
        box=box,
        particle_index=particle_index,
        relative_angle=nominal_angle_change
    )
    position_old = box[particle_index].position
    position_new = box.cube.random_position_inside()
    if position_old.z == 0:
        position_new = Vector(position_new.x, position_new.y)
    move_to = lambda : commands.MoveParticleTo(
        box = box,
        particle_index=particle_index,
        position_new=position_new
    )
    rotate_particle = lambda : commands.RotateParticle(
        box = box,
        particle_index=particle_index,
        relative_angle=nominal_angle_change
    )
    rotate_magnetization = lambda : commands.RotateMagnetization(
        box = box,
        particle_index= particle_index,
        relative_angle=nominal_angle_change
    )
    rescale_single_magnetization = lambda : commands.RescaleMagnetization(
        box = box,
        particle_index= particle_index,
        change_by_factor=change_by_factor
    )
    nuclear_rescale = lambda : commands.NuclearRescale(
        simulation_params=simulation_params,
        change_by_factor=change_by_factor
    )
    magnetic_rescale = lambda : commands.MagneticRescale(
        simulation_params=simulation_params,
        change_by_factor=change_by_factor
    )
    nuclear_magnetic_rescale = lambda : commands.NuclearMagneticRescale(
        simulation_params=simulation_params,
        change_by_factor=change_by_factor
    )
    command_choices = [
        move_by,
        jump_to,
        orbit, 
        move_to, 
        nuclear_magnetic_rescale
        ]
    if not spherical_particle:
        command_choices.append(rotate_particle)
    if magnetic_simulation:
        command_choices.append(rescale_single_magnetization)
        command_choices.append(rotate_magnetization)
    
    return commands.AcceptableCommand(
        base_command= rng.choice(command_choices)(),
        acceptance_scheme=MetropolisAcceptance(
            temperature=temperature
        )
    )

def core_shell_particle_factory(core_radius: float, core_polydispersity: float, core_sld: float, shell_thickness: float, shell_polydispersity: float, shell_sld: float, solvent_sld: float, core_magnetization: float) -> ParticleFactory:
    return lambda : CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=Vector.random_vector_xy(core_magnetization),
        core_radius=rng.normal(loc = core_radius, scale = core_polydispersity * core_radius),
        thickness=rng.normal(loc = shell_thickness, scale = shell_polydispersity * shell_thickness),
        core_sld=core_sld,
        shell_sld=shell_sld,
        solvent_sld=solvent_sld
    )

def generate_particle_factory_from_config_dict(config_dict: dict) -> ParticleFactory:
    particle_type= config_dict.setdefault("particle_type", "CoreShellParticle")
    if particle_type == "CoreShellParticle":
        return core_shell_particle_factory(
            core_radius=config_dict.setdefault("core_radius", 0.0),
            core_polydispersity=config_dict.setdefault("core_polydispersity", 0.0),
            core_sld=config_dict.setdefault("core_sld", 0.0),
            shell_thickness=config_dict.setdefault("shell_thickness", 0.0),
            shell_polydispersity=config_dict.setdefault("shell_polydispersity", 0.0),
            shell_sld=config_dict.setdefault("shell_sld", 0.0),
            solvent_sld=config_dict.setdefault("solvent_sld", 0.0),
            core_magnetization=config_dict.setdefault("core_magnetization", 0.0)
        )


@dataclass
class AnnealingConfig:
    annealing_type: str
    anneal_start_temp: float
    anneal_fall_rate: float
    annealing_stop_cycle_number: int

    def _greedy(self, cycle: int) -> float:
        return 0
    
    def _fast_anneal(self, cycle: int) -> float:
        if cycle > self.annealing_stop_cycle_number:
            return 0
        return self.anneal_start_temp / (cycle + 1)

    def _very_fast_anneal(self, cycle: int) -> float:
        if cycle > self.annealing_stop_cycle_number:
            return 0
        return self.anneal_start_temp * np.exp(-cycle * self.anneal_fall_rate)

    def generate_temperature_function(self) -> Callable[[int], float]:
        d = {
            "greedy" : self._greedy,
            "fast" : self._fast_anneal,
            "very fast" : self._very_fast_anneal
        }
        return self._greedy if self.annealing_type not in d else d[self.annealing_type]


def detector_from_dataframe(data_source: str, data_frames: dict, detector_config: DetectorConfig = None) -> SimulatedDetectorImage:
    if data_source in data_frames:
        data_source_df = data_frames[data_source]
        detector = SimulatedDetectorImage.gen_from_pandas(
            dataframe=data_source_df,
            detector_config=detector_config
        )
        return detector
    if Path(data_source).exists():
        detector = SimulatedDetectorImage.gen_from_txt(
        file_location=Path(data_source),
        detector_config=detector_config
        )
        return detector
    raise Exception(f"Detector named {data_source} could not be found")

def subtract_buffer_intensity(detector: DetectorImage, buffer: Union[float, DetectorImage]) -> DetectorImage:
    if isinstance(buffer, DetectorImage):
        detector.intensity = detector.intensity - buffer.intensity
        detector.intensity_err = np.sqrt(detector.intensity_err**2 + buffer.intensity_err**2) # This is the proper way to do this, since the errors add in quadrature. If the buffer intensity error isn't present, the total error won't change
    elif isinstance(buffer, float):
        detector.intensity = detector.intensity - buffer
    return detector

polarization_dict = {
    "down" : Polarization.SPIN_DOWN,
    "up" : Polarization.SPIN_UP,
    "unpolarized" : Polarization.UNPOLARIZED,
    "unpolarised" : Polarization.UNPOLARIZED,
    "out" : Polarization.UNPOLARIZED
}

def series_to_config_dict(series: pd.Series) -> dict:
    d = {}
    for k, v in series.iteritems():
        add_row_to_dict(d, k, v)
    return d


@dataclass
class DetectorDataConfig:
    data_source: str
    label: str
    detector_config: DetectorConfig = None
    buffer_source: str = ""

    def _generate_buffer(self, data_frames: dict) -> Union[float, DetectorImage]:
        if not self.buffer_source:
            return 0
        if is_float(self.buffer_source):
            return float(self.buffer_source)
        return detector_from_dataframe(
            data_source=self.buffer_source,
            detector_config=self.detector_config,
            data_frames=data_frames
        )

    def generate_detector(self, data_frames: dict) -> SimulatedDetectorImage:
        detector = detector_from_dataframe(
            data_source=self.data_source,
            detector_config=self.detector_config,
            data_frames=data_frames
        )
        buffer = self._generate_buffer(data_frames)
        return subtract_buffer_intensity(detector, buffer)

    @classmethod
    def generate_detectorconfig_from_dict(cls, config_dict: dict):
        detector_config = DetectorConfig(
            detector_distance_in_m=config_dict.setdefault("Detector distance", 0),
            collimation_distance_in_m=config_dict.setdefault("Collimation distance", 0),
            collimation_aperture_area_in_m2=config_dict.setdefault("Collimation aperture", 0),
            sample_aperture_area_in_m2=config_dict.setdefault("Sample aperture", 0),
            detector_pixel_size_in_m=config_dict.setdefault("Detector pixel", 0),
            wavelength_in_angstrom=config_dict.setdefault("Wavelength", 5.0),
            wavelength_spread=config_dict.setdefault("Wavelength Spread", 0.1),
            polarization=polarization_dict[config_dict.setdefault("Polarization", "out")]
        )
        return cls(
            data_source=config_dict.setdefault("Data Source", ""),
            label = config_dict.setdefault("Label", ""),
            detector_config=detector_config,
            buffer_source=config_dict.setdefault("Buffer Source", "")
        )

    @classmethod
    def generate_detectorconfig_list_from_dataframe(cls, dataframe_2: pd.DataFrame) -> List:
        return [cls.generate_detectorconfig_from_dict(series_to_config_dict(row)) for _, row in dataframe_2.iterrows()]


def box_factory(particle_number: int, particle_factory: ParticleFactory, box_template: Tuple[float, float, float]) -> Box:
    dimension_0, dimension_1, dimension_2 = box_template
    return Box(
        particles=[particle_factory() for _ in range(particle_number)],
        cube = Cube(
            dimension_0=dimension_0,
            dimension_1=dimension_1,
            dimension_2=dimension_2
        )
    )

def generate_box_factory(box_template: Tuple[float, float, float], detector_list: List[DetectorImage]) -> BoxFactory:
    dimension_0, dimension_1, dimension_2 = box_template
    if dimension_0 * dimension_1 * dimension_2 == 0:
        dimension_0 = np.max([2 * PI / detector.qx_delta for detector in detector_list])
        dimension_1 = np.max([2 * PI / detector.qy_delta for detector in detector_list])
        dimension_2 = dimension_0
    return lambda particle_number, particle_factory : box_factory(particle_number, particle_factory, (dimension_0, dimension_1, dimension_2))

def generate_save_file(datetime_string: str, output_folder: Path, description: str = "", comment: str = "", file_format: str = "xlsx") -> Path:
    return output_folder / Path(f"{datetime_string}_{description}{comment}.{file_format}")

def generate_file_path_maker(output_folder: Path, description: str = "") -> Callable[[str, str], Path]:
    datetime_format = '%Y%m%d%H%M%S'
    datetime_string = datetime.now().strftime(datetime_format)
    return lambda comment, file_format: generate_save_file(datetime_string, output_folder, description, comment, file_format)


@dataclass
class SimulationConfig:
    simulation_title: str
    particle_factory: ParticleFactory

    nominal_concentration: float
    particle_number: int
    box_number: int

    total_cycles: int
    annealing_config: AnnealingConfig
    nominal_step_size: float

    detector_data_configs: List[DetectorDataConfig]
    
    detector_smearing: bool
    field_direction: str
    box_template: Tuple[float, float, float]
    in_plane: bool = True
    force_log: bool = True
    output_plot_format: str = "none"

    def generate_detector_list(self, data_frames) -> List[DetectorImage]:
        return [detector_data_config.generate_detector(data_frames) for detector_data_config in self.detector_data_configs]

    def generate_box_list(self, detector_list: List[DetectorImage]) -> List[Box]:
        particle_factory = self.particle_factory
        box_factory = generate_box_factory(self.box_template, detector_list )
        if not self.box_number:
            box_volume = box_factory(0, lambda : None).cube.volume
            particle_volume = np.sum([particle_factory().volume for _ in range(self.particle_number)])
            particle_conc = particle_volume / box_volume
            box_number = int(particle_conc / self.nominal_concentration) + 1
            average_particle_volume = particle_volume / self.particle_number
            particle_number_per_box = int(self.nominal_concentration * box_volume / average_particle_volume)
            return [box_factory(particle_number_per_box, particle_factory) for _ in range(box_number)]
        if not self.particle_number:
            box_volume = box_factory(0, lambda : None).cube.volume
            particle_volume = np.average([particle_factory().volume for _ in range(100)])
            particle_number_per_box = int(self.nominal_concentration / (particle_volume / box_volume))
            return [box_factory(particle_number_per_box, particle_factory) for _ in range(self.box_number)]
        particle_number_per_box = int(self.particle_number / self.box_number)
        return [box_factory(particle_number_per_box, particle_factory) for _ in range(self.box_number)]

    def generate_save_file_maker(self, output_folder: Path) -> Callable[[str, str], Path]:
        return generate_file_path_maker(output_folder=output_folder, description=self.simulation_title)
        #return generate_save_file(output_folder=output_folder, description=self.simulation_title, file_format="xlsx")

    def generate_scattering_simulation(self, detector_list: List[DetectorImage], box_list: List[Box]) -> ScatteringSimulation:
        qxqy_list = qxqy_array_list_from_detector_list(detector_list, range_factor=RANGE_FACTOR)
        fitter = Fitter2D.generate_standard_fitter(
            simulated_detectors=detector_list,
            box_list=box_list,
            qxqy_list=qxqy_list
        ) if self.detector_smearing else Fitter2D.generate_no_smear_fitter(
            simulated_detectors=detector_list,
            box_list=box_list
        )
        return ScatteringSimulation(fitter = fitter, simulation_params=box_simulation_params_factory())

    def generate_controller(self, simulation: ScatteringSimulation, box_list: List[Box]) -> Controller:
        temperature_function = self.annealing_config.generate_temperature_function()
        ledger = [
            commands.SetSimulationParams(simulation.simulation_params, change_to_factors = simulation.simulation_params.values),
            ]
        for box in box_list:
            box.force_inside_box(in_plane=self.in_plane)
            for particle_index, _ in enumerate(box.particles):
                ledger.append(
                    commands.SetParticleState.gen_from_particle(box, particle_index)
                )
        
        for cycle in range(self.total_cycles):
            box_ledger = []
            temperature = temperature_function(cycle)
            for box_index, box in enumerate(box_list):
                for particle_index, particle in enumerate(box.particles):
                    acceptable_command = command_factory(
                        nominal_step_size=self.nominal_step_size,
                        box = box,
                        particle_index=particle_index,
                        temperature=temperature,
                        simulation_params=simulation.simulation_params,
                        spherical_particle=particle.is_spherical(),
                        magnetic_simulation=box.is_magnetic()
                    )
                    acceptable_command.update_loggable_data(
                        {"Cycle": cycle, "Box index": box_index}
                    )
                    box_ledger.append(acceptable_command)
            rng.shuffle(box_ledger)
            ledger += box_ledger    

        return Controller(
            ledger=ledger
        )

    def generate_simulator(self, controller: Controller, simulation: ScatteringSimulation, box_list: List[Box]) -> Simulator:
        
        return MemorizedSimulator(
            controller=controller,
            evaluator=MonteCarloEvaluator(
                simulation=simulation,
                viewer=CLIViewer()
            ),
            simulation=simulation,
            box_list=box_list
        )

    @classmethod
    def gen_from_dataframes(cls, dataframes: dict):
        dataframe, dataframe_2 = list(dataframes.values())[:2]
        config_dict = dataframe_to_config_dict(dataframe) 
        total_cycles = config_dict.setdefault("total_cycles", 0)
        annealing_stop_cycle_as_read = config_dict.setdefault("annealing_stop_cycle_number", total_cycles / 2)
        annealing_config = AnnealingConfig(
            annealing_type=config_dict.setdefault("annealing_type", "greedy").strip().lower(),
            anneal_start_temp=config_dict.setdefault("anneal_start_temp", 0.0),
            anneal_fall_rate=config_dict.setdefault("anneal_fall_rate",0.1),
            annealing_stop_cycle_number=annealing_stop_cycle_as_read
        )
        detector_data_configs = [DetectorDataConfig.generate_detectorconfig_from_dict(config_dict)] if (config_dict.setdefault("Data Source", "")) else DetectorDataConfig.generate_detectorconfig_list_from_dataframe(dataframe_2)
        return cls(
            simulation_title = config_dict.setdefault("simulation_title", ""),
            particle_factory = generate_particle_factory_from_config_dict(config_dict),
            nominal_concentration=config_dict.setdefault("nominal_concentration", 0.0),
            particle_number=config_dict.setdefault("particle_number", 0),
            box_number=config_dict.setdefault("box_number", 0),
            total_cycles = total_cycles,
            annealing_config=annealing_config,
            nominal_step_size = config_dict.setdefault("core_radius", 100) / 2, # make this something more general later
            detector_data_configs=detector_data_configs,
            detector_smearing=config_dict.setdefault("detector_smearing", True),
            field_direction=config_dict.setdefault("field_direction", "Y"),
            force_log = config_dict.setdefault("force_log_file", True),
            output_plot_format = config_dict.setdefault("output_plot_format", "NONE").lower(),
            box_template = (
                config_dict.setdefault("box_dimension_1", 0),
                config_dict.setdefault("box_dimension_2", 0),
                config_dict.setdefault("box_dimension_3", 0)
            )
        )
    







    

        



