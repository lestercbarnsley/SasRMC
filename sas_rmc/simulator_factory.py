
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from .array_cache import array_cache
from . import commands
from .acceptance_scheme import MetropolisAcceptance
from .viewer import CLIViewer
from .scattering_simulation import Fitter2D, ScatteringSimulation, SimulationParams
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

Reader = Callable[[str, Optional[Type], Optional[Any]], Any]
ParticleFactory = Callable[[], Particle]
BoxFactory = Callable[[int,ParticleFactory], Box]

DEFAULT_VAL_TYPES =  {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    Polarization: Polarization.UNPOLARIZED
}

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError: # This is the only time I would ever except an error, why on earth does Python not have a built-in isfloat method?
        return False

def add_row_to_dict(d: dict, param_name: str, param_value: str) -> None:
    if not param_name.strip():
        return
    if r'#' in param_name.strip():
        return
    if not param_value.strip():
        return
    d[param_name] = param_value.strip()

def dict_reader(config_dict: dict, key: str, t: Type = float, default_value: Any = None) -> Any:
    type_in_default_dict = lambda : DEFAULT_VAL_TYPES[t] if t in DEFAULT_VAL_TYPES else 0 # lambda to delay calculation
    default_value_to_use = default_value if default_value else type_in_default_dict()
    print('key', key, 'value', 'not in dict' if key not in config_dict else config_dict[key])
    if (key not in config_dict) or not config_dict[key]:
        if key == "Data Source":
            print("Data source",t(default_value_to_use))
        return t(default_value_to_use)
    if key == "Data Source":
        print("Data source",t(config_dict[key]))
    return t(config_dict[key])

def dataframe_to_config_dict_with_reader(dataframe: pd.DataFrame) -> Tuple[dict, Reader]:
    config_dict = dict()
    for _, row in dataframe.iterrows():
        param_name = row.iloc[0]
        param_value = row.iloc[1]
        add_row_to_dict(config_dict, param_name, param_value)
    return config_dict, lambda key, t = float, default_value = 0 : dict_reader(config_dict, key, t, default_value)

def different_random_int(n: int, number_to_avoid: int) -> int:
    for _ in range(200000):
        x = rng.choice(range(n))
        if x != number_to_avoid:
            return x
    return -1

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
    print('id qx', id(qX))
    print('id qy', id(qY))
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
    

@dataclass
class ParticleConfig:
    particle_type: str
    core_radius: float
    core_polydispersity: float
    core_sld: float
    shell_thickness: float
    shell_polydispersity: float
    shell_sld: float
    solvent_sld: float
    core_magnetization: float

    def generate_particle_factory(self) -> ParticleFactory:
        factory = lambda : CoreShellParticle.gen_from_parameters(
            position = Vector.null_vector(),
            magnetization=Vector.random_vector_xy(self.core_magnetization),
            core_radius=rng.normal(loc = self.core_radius, scale = self.core_polydispersity * self.core_radius),
            thickness=rng.normal(loc = self.shell_thickness, scale = self.shell_polydispersity * self.shell_thickness),
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )
        return factory

    @classmethod
    def gen_from_d_reader(cls, d_reader: Reader):
        return cls(
            particle_type= d_reader("particle_type", str),
            core_radius=d_reader("core_radius"),
            core_polydispersity=d_reader("core_polydispersity"),
            core_sld=d_reader("core_sld"),
            shell_thickness=d_reader("shell_thickness"),
            shell_polydispersity=d_reader("shell_polydispersity"),
            shell_sld=d_reader("shell_sld"),
            solvent_sld=d_reader("solvent_sld"),
            core_magnetization=d_reader("core_magnetization")
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

polarization_dict = {
    "down" : Polarization.SPIN_DOWN,
    "up" : Polarization.SPIN_UP,
    "unpolarized" : Polarization.UNPOLARIZED,
    "unpolarised" : Polarization.UNPOLARIZED,
    "out" : Polarization.UNPOLARIZED
}


@dataclass
class DetectorDataConfig:
    data_source: str
    label: str
    detector_config: DetectorConfig = None
    buffer_source: str = ""

    def _generate_buffer_array(self, data_frames: dict) -> Union[np.ndarray, float]:
        if not self.buffer_source:
            return 0
        if is_float(self.buffer_source):
            return float(self.buffer_source)
        return detector_from_dataframe(
            data_source=self.buffer_source,
            detector_config=self.detector_config,
            data_frames=data_frames
        ).intensity

    def generate_detector(self, data_frames: dict) -> SimulatedDetectorImage:
        detector = detector_from_dataframe(
            data_source=self.data_source,
            detector_config=self.detector_config,
            data_frames=data_frames
        )
        buffer_array = self._generate_buffer_array(data_frames)
        detector.intensity = detector.intensity - buffer_array
        return detector

    @classmethod
    def generate_detectorconfig_from_dict(cls, d_reader: Reader):
        detector_config = DetectorConfig(
            detector_distance_in_m=d_reader("Detector distance"),
            collimation_distance_in_m=d_reader("Collimation distance"),
            collimation_aperture_area_in_m2=d_reader("Collimation aperture"),
            sample_aperture_area_in_m2=d_reader("Sample aperture"),
            detector_pixel_size_in_m=d_reader("Detector pixel"),
            wavelength_in_angstrom=d_reader("Wavelength"),
            wavelength_spread=d_reader("Wavelength Spread"),
            polarization=polarization_dict[d_reader("Polarization", str, default_value = "out")]
        )
        return cls(
            data_source=d_reader("Data Source", str),
            label = d_reader("Label", str),
            detector_config=detector_config,
            buffer_source=d_reader("Buffer Source", str)
        )

    @classmethod
    def generate_detectorconfig_list_from_dataframe(cls, dataframe_2: pd.DataFrame) -> List:
        '''def get_row_reader(df_row, key, t = float, default_value = None):
            df_dict = df_row.to_dict()
            return dict_reader(df_dict, key, t, default_value=default_value)'''
        return [cls.generate_detectorconfig_from_dict(lambda key, t = float, default_value = None : dict_reader(row.to_dict(), key, t, default_value)) for _, row in dataframe_2.iterrows()]


def generate_box_factory(box_template: Tuple[float, float, float],  detector_list: List[DetectorImage]) -> BoxFactory:
    dimension_0, dimension_1, dimension_2 = box_template
    if dimension_0 * dimension_1 * dimension_2 == 0:
        dimension_0 = np.max([2 * PI / detector.qx_delta for detector in detector_list])
        dimension_1 = np.max([2 * PI / detector.qy_delta for detector in detector_list])
        dimension_2 = dimension_0
    def box_factory(particle_number: int, particle_factory: ParticleFactory) -> Box:
        return Box(
            particles=[particle_factory() for _ in range(particle_number)],
            cube = Cube(
                dimension_0=dimension_0,
                dimension_1=dimension_1,
                dimension_2=dimension_2
            )
        )
    return box_factory
    
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
            
@dataclass
class SimulationConfig:
    simulation_title: str
    particle_config: ParticleConfig

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

    def generate_detector_list(self, data_frames) -> List[DetectorImage]:
        return [detector_data_config.generate_detector(data_frames) for detector_data_config in self.detector_data_configs]

    def generate_box_list(self, detector_list: List[DetectorImage]) -> List[Box]:
        particle_factory = self.particle_config.generate_particle_factory()
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

    def generate_save_file(self, output_folder: Path):
        datetime_format = '%Y%m%d%H%M%S'
        return output_folder / Path(f"{datetime.now().strftime(datetime_format)}_{self.simulation_title}.xlsx")

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
        return ScatteringSimulation(fitter = fitter)

    def generate_controller(self, simulation: ScatteringSimulation, box_list: List[Box]) -> Controller:
        temperature_function = self.annealing_config.generate_temperature_function()
        ledger = [
            commands.NuclearScale(simulation.simulation_params, change_to_factor=simulation.simulation_params.rescale_factor),
            commands.MagneticScale(simulation.simulation_params, change_to_factor=simulation.simulation_params.magnetic_rescale),
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
                        magnetic_simulation=bool(self.particle_config.core_magnetization)
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
        dataframe = list(dataframes.values())[0]
        dataframe_2 = list(dataframes.values())[1]
        config_dict, dict_reader = dataframe_to_config_dict_with_reader(dataframe) # ironically I don't need the config dict, because the reader func lets me acces everything
        total_cycles = dict_reader("total_cycles", int)
        annealing_stop_cycle_as_read = dict_reader("annealing_stop_cycle_number", int, total_cycles / 2)
        annealing_config = AnnealingConfig(
            annealing_type=dict_reader("annealing_type", str).strip().lower(),
            anneal_start_temp=dict_reader("anneal_start_temp"),
            anneal_fall_rate=dict_reader("anneal_fall_rate"),
            annealing_stop_cycle_number=annealing_stop_cycle_as_read
        )
        detector_data_configs = [DetectorDataConfig.generate_detectorconfig_from_dict(dict_reader)] if dict_reader("Data Source", str, "") else DetectorDataConfig.generate_detectorconfig_list_from_dataframe(dataframe_2)
        return cls(
            simulation_title = dict_reader("simulation_title", str),
            particle_config = ParticleConfig.gen_from_d_reader(dict_reader),
            nominal_concentration=dict_reader("nominal_concentration"),
            particle_number=dict_reader("particle_number", int),
            box_number=dict_reader("box_number", int),
            total_cycles = total_cycles,
            annealing_config=annealing_config,
            nominal_step_size = dict_reader("core_radius") / 2, # make this something more general later
            detector_data_configs=detector_data_configs,
            detector_smearing=truth_dict[dict_reader("detector_smearing", str)],
            field_direction=dict_reader("field_direction", str),
            force_log = truth_dict[dict_reader("force_log_file", str)],
            box_template = (
                dict_reader("box_dimension_1"),
                dict_reader("box_dimension_2"),
                dict_reader("box_dimension_3")
            )
        )
    







    

        



