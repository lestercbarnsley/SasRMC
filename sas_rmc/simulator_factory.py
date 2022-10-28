
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple, Type, Union

import numpy as np
import pandas as pd

from . import commands, constants
from .converters import dict_to_particle
from .acceptance_scheme import MetropolisAcceptance
from .viewer import CLIViewer
from .scattering_simulation import MAGNETIC_RESCALE, NUCLEAR_RESCALE, ScatteringSimulation, SimulationParam, SimulationParams
from .simulator import MemorizedSimulator, MonteCarloEvaluator, Simulator
from .vector import Vector, VectorSpace
from .particles import CoreShellParticle, Dumbbell, Particle
from .box_simulation import Box
from .detector import DetectorImage, Polarization, DetectorConfig, SimulatedDetectorImage
from .shapes.shapes import Cube
from .controller import Controller
from .result_calculator import AnalyticalCalculator, NumericalCalculator, ResultCalculator
from .fitter import Fitter2D

rng = np.random.default_rng()
PI = constants.PI
RANGE_FACTOR = 1.2
RESOLUTION_FACTOR = 1.0
NUMERICAL_RANGE_FACTOR = 1.05


ParticleFactory = Callable[[], Particle]
BoxFactory = Callable[[int,ParticleFactory], Box]
  
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

def box_simulation_params_factory(starting_rescale: float = 1.0, starting_magnetic_rescale: float = 1.0) -> SimulationParams:
    params = [
        SimulationParam(value = starting_rescale, name = NUCLEAR_RESCALE, bounds=(0, np.inf)), 
        SimulationParam(value = starting_magnetic_rescale, name = MAGNETIC_RESCALE, bounds=(0, np.inf))
        ]
    return SimulationParams(params = params)


def command_factory(nominal_step_size: float, box: Box, particle_index: int, temperature: float, simulation_params: SimulationParams, spherical_particle: bool = False, magnetic_simulation: bool = False) -> commands.AcceptableCommand:
    nominal_angle_change = PI/8
    nominal_rescale= 0.02
    particle_was_in_plane = box[particle_index].position.z == 0
    change_by_factor = rng.normal(loc = 1.0, scale = nominal_rescale)
    position_delta_size = rng.normal(loc = 0.0, scale = nominal_step_size)
    position_delta = (Vector.random_vector_xy if particle_was_in_plane else Vector.random_vector)(position_delta_size)
    actual_angle_change = rng.normal(loc = 0.0, scale = nominal_angle_change)
    massive_angle_change = rng.uniform(low = -PI, high = PI)
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
        relative_angle=actual_angle_change
    )
    put_in_plane = lambda vector : Vector(vector.x, vector.y, z = 0 if particle_was_in_plane else box[particle_index].position.z)
    position_new = put_in_plane(box.cube.random_position_inside())
    '''
    if particle_was_in_plane:
        position_new = Vector(position_new.x, position_new.y)
    else:
        position_new = Vector(position_new.x, position_new.y, box[particle_index].position.z)'''
    move_to = lambda : commands.MoveParticleTo(
        box = box,
        particle_index=particle_index,
        position_new=position_new
    )
    rotate_particle = lambda : commands.RotateParticle(
        box = box,
        particle_index=particle_index,
        relative_angle=actual_angle_change
    )
    rotate_magnetization = lambda : commands.RotateMagnetization(
        box = box,
        particle_index= particle_index,
        relative_angle=actual_angle_change
    )
    rotate_magnetization_large = lambda : commands.RotateMagnetization(
        box = box,
        particle_index= particle_index,
        relative_angle=massive_angle_change
    )
    rescale_single_magnetization = lambda : commands.RescaleMagnetization(
        box = box,
        particle_index= particle_index,
        change_by_factor=change_by_factor
    )
    flip_single_magnetization = lambda : commands.FlipMagnetization(
        box = box,
        particle_index= particle_index
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
        command_choices.extend([
            #rescale_single_magnetization, 
            rotate_magnetization,
            rotate_magnetization_large,
            flip_single_magnetization,
            ])
        
    
    return commands.AcceptableCommand(
        base_command= rng.choice(command_choices)(),
        acceptance_scheme=MetropolisAcceptance(
            temperature=temperature
        )
    )

def polydisperse_parameter(loc: float, polyd: float, dispersity_fn: Callable[[float, float], float] = None) -> float:
    poly_fn = dispersity_fn if dispersity_fn else (lambda l, s: rng.normal(loc = l, scale = s)) # I only want to write out the lambda expression like this so I can be explicit about the kwargs
    return poly_fn(loc, loc * polyd)

def core_shell_particle_factory(config_dict: dict) -> ParticleFactory:#core_radius: float, core_polydispersity: float, core_sld: float, shell_thickness: float, shell_polydispersity: float, shell_sld: float, solvent_sld: float, core_magnetization: float) -> ParticleFactory:
    core_magnetization = config_dict.get("core_magnetization", 0.0)
    core_radius = config_dict.get("core_radius", 0.0)
    core_polydispersity = config_dict.get("core_polydispersity", 0.0)
    shell_thickness = config_dict.get("shell_thickness", 0.0)
    shell_polydispersity = config_dict.get("shell_polydispersity", 0.0)
    core_sld = config_dict.get("core_sld", 0.0)
    shell_sld = config_dict.get("shell_sld", 0.0)
    solvent_sld = config_dict.get("solvent_sld", 0.0)
    return lambda : CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=Vector.random_vector_xy(core_magnetization),#Vector(0, core_magnetization, 0),#
        core_radius=polydisperse_parameter(loc = core_radius, polyd = core_polydispersity),
        thickness=polydisperse_parameter(loc = shell_thickness, polyd=shell_polydispersity),
        core_sld=core_sld,
        shell_sld=shell_sld,
        solvent_sld=solvent_sld
    )

def dumbbell_particle_factory(config_dict: dict) -> ParticleFactory:
    core_radius = config_dict.get("core_radius", 0.0)
    core_polydispersity = config_dict.get("core_polydispersity", 0.0)
    core_sld= config_dict.get("core_sld", 0.0)
    seed_radius = config_dict.get("seed_radius", 0.0)
    seed_polydispersity = config_dict.get("seed_polydispersity", 0.0)
    seed_sld = config_dict.get("seed_sld", 0.0)
    shell_thickness = config_dict.get("shell_thickness", 0.0)
    shell_polydispersity = config_dict.get("shell_polydispersity", 0.0)
    shell_sld = config_dict.get("shell_sld", 0.0)
    solvent_sld = config_dict.get("solvent_sld", 0.0)
    core_magnetization = config_dict.get("core_magnetization", 0.0)
    seed_magnetization= config_dict.get("seed_magnetization", 0.0)
    return lambda : Dumbbell.gen_from_parameters(
        core_radius=polydisperse_parameter(loc = core_radius, polyd=core_polydispersity),
        seed_radius=polydisperse_parameter(loc = seed_radius, polyd = seed_polydispersity),
        shell_thickness=polydisperse_parameter(loc = shell_thickness, polyd = shell_polydispersity),
        core_sld = core_sld,
        seed_sld = seed_sld,
        shell_sld=shell_sld,
        solvent_sld=solvent_sld,
        position = Vector.null_vector(),
        orientation=Vector.random_vector_xy(),
        core_magnetization = Vector.random_vector_xy(core_magnetization),
        seed_magnetization=Vector.random_vector_xy(seed_magnetization)
    )

def generate_particle_factory_from_config_dict(config_dict: dict) -> ParticleFactory:
    particle_type= config_dict.get("particle_type", "CoreShellParticle")
    if particle_type == CoreShellParticle.__name__:
        return core_shell_particle_factory(config_dict)
    if particle_type == Dumbbell.__name__:
        return dumbbell_particle_factory(config_dict)
    



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
    detector.shadow_factor = detector.shadow_factor * (detector.intensity > 0)
    return detector

polarization_dict = {
    "down" : Polarization.SPIN_DOWN,
    "up" : Polarization.SPIN_UP,
    "unpolarized" : Polarization.UNPOLARIZED,
    "unpolarised" : Polarization.UNPOLARIZED,
    "out" : Polarization.UNPOLARIZED
}

def get_polarization(polarization_str: str) -> Polarization:
    try:
        polarization = Polarization(polarization_str)
        return polarization
    except ValueError:
        return polarization_dict.get(polarization_str, Polarization.UNPOLARIZED)

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
            detector_distance_in_m=config_dict.get("Detector distance", 0),
            collimation_distance_in_m=config_dict.get("Collimation distance", 0),
            collimation_aperture_area_in_m2=config_dict.get("Collimation aperture", 0),
            sample_aperture_area_in_m2=config_dict.get("Sample aperture", 0),
            detector_pixel_size_in_m=config_dict.get("Detector pixel", 0),
            wavelength_in_angstrom=config_dict.get("Wavelength", 5.0),
            wavelength_spread=config_dict.get("Wavelength Spread", 0.1),
            polarization=get_polarization(config_dict.get("Polarization", "out")),
        )
        return cls(
            data_source=config_dict.get("Data Source", ""),
            label = config_dict.get("Label", ""),
            detector_config=detector_config,
            buffer_source=config_dict.get("Buffer Source", "")
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

def generate_box_factory(box_template: Tuple[float, float, float], detector_list: List[DetectorImage], delta_qxqy_strategy: Callable[[DetectorImage], Tuple[float, float]] = None) -> BoxFactory:
    delta_qxqy_strategy = delta_qxqy_strategy if delta_qxqy_strategy is not None else (lambda d : d.delta_qxqy_from_detector())
    dimension_0, dimension_1, dimension_2 = box_template
    if dimension_0 * dimension_1 * dimension_2 == 0:
        dimension_0 = np.max([2 * PI / delta_qxqy_strategy(detector)[0] for detector in detector_list])
        dimension_1 = np.max([2 * PI / delta_qxqy_strategy(detector)[1] for detector in detector_list])
        dimension_2 = dimension_0
    return lambda particle_number, particle_factory : box_factory(particle_number, particle_factory, (dimension_0, dimension_1, dimension_2))

def generate_save_file(datetime_string: str, output_folder: Path, description: str = "", comment: str = "", file_format: str = "xlsx") -> Path:
    return output_folder / Path(f"{datetime_string}_{description}{comment}.{file_format}")

def generate_file_path_maker(output_folder: Path, description: str = "") -> Callable[[str, str], Path]:
    datetime_format = '%Y%m%d%H%M%S'
    datetime_string = datetime.now().strftime(datetime_format)
    return lambda comment, file_format: generate_save_file(datetime_string, output_folder, description, comment, file_format)

def qxqy_from_detector(detector: DetectorImage, range_factor: float, resolution_factor: float, delta_qxqy_strategy: Callable[[DetectorImage], Tuple[float, float]] = None):
    detector_qx, detector_qy = detector.qX, detector.qY
    delta_qxqy_strategy = delta_qxqy_strategy if delta_qxqy_strategy is not None else (lambda d : d.delta_qxqy_from_detector())
    detector_delta_qx, detector_delta_qy = delta_qxqy_strategy(detector)
    line_maker = lambda starting_q_min, starting_q_max, starting_q_step : np.arange(start = range_factor * starting_q_min, stop = +range_factor * starting_q_max, step = starting_q_step / resolution_factor)
    qx_linear = line_maker(np.min(detector_qx), np.max(detector_qx), detector_delta_qx)
    qy_linear = line_maker(np.min(detector_qy), np.max(detector_qy), detector_delta_qy)
    qx, qy = np.meshgrid(qx_linear, qy_linear)
    return qx, qy

def analytical_calculator_maker(detector: DetectorImage, range_factor: float, resolution_factor: float):
    qx, qy = qxqy_from_detector(detector, range_factor, resolution_factor)
    return AnalyticalCalculator(qx, qy)

def numerical_calculator_maker(detector: DetectorImage, particle_factory: ParticleFactory, range_factor: float, resolution_factor: float):
    biggest_dimension_0, biggest_dimension_1, biggest_dimension_2 = 0, 0, 0
    for _ in range(10000):
        particle = particle_factory()
        for shape in particle.shapes:
            d_0, d_1, d_2 = shape.dimensions
            biggest_dimension_0 = np.max(NUMERICAL_RANGE_FACTOR *  d_0, biggest_dimension_0)
            biggest_dimension_1 = np.max(NUMERICAL_RANGE_FACTOR *  d_1, biggest_dimension_1)
            biggest_dimension_2 = np.max(NUMERICAL_RANGE_FACTOR *  d_2, biggest_dimension_2)
    vector_space_resolution = PI / np.max(detector.q)
    vector_space = VectorSpace.gen_from_bounds(
        x_min = -biggest_dimension_0 / 2, x_max = biggest_dimension_0 / 2, x_num = int(biggest_dimension_0 / vector_space_resolution),
        y_min = -biggest_dimension_1 / 2, y_max = biggest_dimension_1 / 2, y_num = int(biggest_dimension_1 / vector_space_resolution),
        z_min = -biggest_dimension_2 / 2, z_max = biggest_dimension_2 / 2, z_num = int(biggest_dimension_2 / vector_space_resolution)
    )
    qx, qy = qxqy_from_detector(detector, range_factor, resolution_factor)
    return NumericalCalculator(qx, qy, vector_space)
    
def generate_result_calculator_maker(particle_factory: ParticleFactory, calculator_type: str = "Analytical", range_factor: float = RANGE_FACTOR, resolution_factor: float = RESOLUTION_FACTOR) -> Callable[[DetectorImage], ResultCalculator]:
    if calculator_type == "Analytical":
        return lambda detector : analytical_calculator_maker(detector, range_factor=range_factor, resolution_factor=resolution_factor)
    if calculator_type == "Numerical":
        return lambda detector : numerical_calculator_maker(detector, particle_factory, range_factor=range_factor, resolution_factor=resolution_factor)
    raise Exception(f"Your entered calculator type: {calculator_type}, isn't a valid entry for 'calculator_type'")


@dataclass
class SimulationConfig:
    simulation_title: str
    particle_factory: ParticleFactory
    result_calculator_maker : Callable[[DetectorImage], ResultCalculator]

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
        fitter = Fitter2D.generate_standard_fitter(
            detector_list=detector_list,
            box_list = box_list,
            result_calculator_maker=self.result_calculator_maker,
            smearing=self.detector_smearing
        )
        box_concentration = np.sum([p.volume for p in box_list[0]]) / box_list[0].volume
        starting_factor = self.nominal_concentration / box_concentration if self.nominal_concentration else 1.0
        return ScatteringSimulation(fitter = fitter, simulation_params=box_simulation_params_factory(starting_rescale=starting_factor, starting_magnetic_rescale=starting_factor))

    def generate_controller(self, simulation: ScatteringSimulation, box_list: List[Box]) -> Controller:
        temperature_function = self.annealing_config.generate_temperature_function()
        ledger = [
            commands.SetSimulationParams(simulation.simulation_params, change_to_factors = simulation.simulation_params.values),
            ]
        for box in box_list:
            box.force_inside_box(in_plane=True)
            for particle_index, particle in enumerate(box.particles):
                if not self.in_plane:
                    box.particles[particle_index] = particle.set_position(particle.position + Vector(0,0,1e-3))
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
    def gen_from_dataframes(cls, config_dict: dict, dataframe_2: pd.DataFrame):
        total_cycles = config_dict.get("total_cycles", 0)
        annealing_stop_cycle_as_read = config_dict.get("annealing_stop_cycle_number", total_cycles / 2)
        annealing_config = AnnealingConfig(
            annealing_type=config_dict.get("annealing_type", "greedy").strip().lower(),
            anneal_start_temp=config_dict.get("anneal_start_temp", 0.0),
            anneal_fall_rate=config_dict.get("anneal_fall_rate",0.1),
            annealing_stop_cycle_number=annealing_stop_cycle_as_read
        )
        detector_data_configs = [DetectorDataConfig.generate_detectorconfig_from_dict(config_dict)] if (config_dict.get("Data Source", "")) else DetectorDataConfig.generate_detectorconfig_list_from_dataframe(dataframe_2)
        particle_factory = generate_particle_factory_from_config_dict(config_dict)
        return cls(
            simulation_title = config_dict.get("simulation_title", ""),
            particle_factory = generate_particle_factory_from_config_dict(config_dict),
            result_calculator_maker = generate_result_calculator_maker(particle_factory, config_dict.get("result_calculator", "Analytical")),
            nominal_concentration=config_dict.get("nominal_concentration", 0.0),
            particle_number=config_dict.get("particle_number", 0),
            box_number=config_dict.get("box_number", 0),
            total_cycles = total_cycles,
            annealing_config=annealing_config,
            nominal_step_size = config_dict.get("core_radius", 100) / 2, # make this something more general later
            detector_data_configs=detector_data_configs,
            detector_smearing=config_dict.get("detector_smearing", True),
            field_direction=config_dict.get("field_direction", "Y"),
            force_log = config_dict.get("force_log_file", True),
            in_plane = config_dict.get("in_plane", False),
            output_plot_format = config_dict.get("output_plot_format", "PDF").lower(),
            box_template = (
                config_dict.get("box_dimension_1", 0),
                config_dict.get("box_dimension_2", 0),
                config_dict.get("box_dimension_3", 0)
            )
        )

def box_from_detector(detector_list: DetectorImage, particle_list: List[Particle], delta_qxqy_strategy: Callable[[DetectorImage], Tuple[float, float]] = None) -> Box:
    delta_qxqy_strategy = delta_qxqy_strategy if delta_qxqy_strategy is not None else (lambda d : d.delta_qxqy_from_detector())
    dimension_0 = np.max([2 * PI / delta_qxqy_strategy(detector)[0] for detector in detector_list])
    dimension_1 = np.max([2 * PI / delta_qxqy_strategy(detector)[1] for detector in detector_list])
    dimension_2 = dimension_0
    return Box(
        particles=particle_list,
        cube = Cube(
            dimension_0=dimension_0,
            dimension_1=dimension_1,
            dimension_2=dimension_2
        )
    )

def get_final_simulation_params(simulation_data: pd.DataFrame) -> Tuple[float, float]:
    nuclear_rescale, magnetic_rescale = 1,1
    valid_non_zero_float = lambda v: is_float(v) and float(v)
    for _, simulation_row in simulation_data.iterrows():
        if simulation_row['Acceptable Move'] == 'TRUE' or simulation_row['Acceptable Move'] == True:
            #print(simulation_row['Acceptable Move'])
            current_nuclear_rescale = simulation_row[NUCLEAR_RESCALE]
            if valid_non_zero_float(current_nuclear_rescale):
                nuclear_rescale = float(current_nuclear_rescale)
            current_magnetic_rescale = simulation_row[MAGNETIC_RESCALE]
            if valid_non_zero_float(current_magnetic_rescale):
                magnetic_rescale = float(current_magnetic_rescale)
    return nuclear_rescale, magnetic_rescale


@dataclass
class SimulationReloader(SimulationConfig):
    data_frames: dict = field(default_factory = dict)

    def generate_detector_list(self, data_frames = None) -> List[DetectorImage]:
        sdfs = self.data_frames
        detector_list = []
        for i in range(100_000):
            s = f"Final detector image {i}"
            if s in sdfs:
                detector_list.append(SimulatedDetectorImage.gen_from_pandas(sdfs[s]))
            else:
                break
        return detector_list

    def generate_box_list(self, detector_list: List[DetectorImage]) -> List[Box]:
        box_list = []
        for i in range(100_000):
            s = f"Box {i} Final Particle States"
            if s in self.data_frames:
                particles = [dict_to_particle(row.to_dict()) for _, row in self.data_frames[s].iterrows()]
                box = box_from_detector(detector_list, particles)
                box_list.append(box)
            else:
                break
        return box_list

    def generate_controller(self, simulation: ScatteringSimulation, box_list: List[Box]) -> Controller:
        super_controller = super().generate_controller(simulation, box_list)
        simulation_data = self.data_frames['Simulation Data']
        nuclear_rescale, magnetic_rescale = get_final_simulation_params(simulation_data)
        simulation.simulation_params.set_value(NUCLEAR_RESCALE, nuclear_rescale)
        simulation.simulation_params.set_value(MAGNETIC_RESCALE, magnetic_rescale)
        first_command = commands.SetSimulationParams(simulation.simulation_params, change_to_factors = simulation.simulation_params.values)
        super_controller.ledger[0] = first_command
        return super_controller

    @classmethod
    def gen_from_dataframes(cls, config_dict: dict, dataframe_2: pd.DataFrame):
        total_cycles = config_dict.get("total_cycles", 0)
        annealing_stop_cycle_as_read = config_dict.get("annealing_stop_cycle_number", total_cycles / 2)
        annealing_config = AnnealingConfig(
            annealing_type=config_dict.get("annealing_type", "greedy").strip().lower(),
            anneal_start_temp=config_dict.get("anneal_start_temp", 0.0),
            anneal_fall_rate=config_dict.get("anneal_fall_rate",0.1),
            annealing_stop_cycle_number=annealing_stop_cycle_as_read
        )
        detector_data_configs = [DetectorDataConfig.generate_detectorconfig_from_dict(config_dict)] if (config_dict.get("Data Source", "")) else DetectorDataConfig.generate_detectorconfig_list_from_dataframe(dataframe_2)
        simulation_data_location = config_dict.get("log_file_source")
        simulation_data_frames = pd.read_excel(
           simulation_data_location,
           dtype = None,
           sheet_name = None,
           keep_default_na=False,
        )
        return cls(
            simulation_title = config_dict.get("simulation_title", ""),
            data_frames=simulation_data_frames,
            particle_factory = None,
            nominal_concentration=config_dict.get("nominal_concentration", 0.0),
            particle_number=config_dict.get("particle_number", 0),
            box_number=config_dict.get("box_number", 0),
            total_cycles = total_cycles,
            annealing_config=annealing_config,
            nominal_step_size = config_dict.get("core_radius", 100) / 2, # make this something more general later
            detector_data_configs=detector_data_configs,
            detector_smearing=config_dict.get("detector_smearing", True),
            field_direction=config_dict.get("field_direction", "Y"),
            force_log = config_dict.get("force_log_file", True),
            output_plot_format = config_dict.get("output_plot_format", "PDF").lower(),
            box_template = (
                config_dict.get("box_dimension_1", 0),
                config_dict.get("box_dimension_2", 0),
                config_dict.get("box_dimension_3", 0)
            )
        )    


def gen_config_from_dataframes(data_frames: dict) -> SimulationConfig:
    dataframe, dataframe_2 = list(data_frames.values())[:2]
    config_dict = dataframe_to_config_dict(dataframe)
    t = SimulationReloader if config_dict["particle_type"] == "Reload old simulation" else SimulationConfig
    return t.gen_from_dataframes(config_dict, dataframe_2)
        

        



