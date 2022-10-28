#%%
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..shapes.shapes import Cube
from ..box_simulation import Box
from ..detector import DetectorImage
from .particle_factory import ParticleFactory
from .. import constants

PI = constants.PI

@dataclass
class BoxFactory(ABC):

    @abstractmethod
    def create_box(self, particle_factory: ParticleFactory, total_particles: int) -> Box:
        pass


@dataclass
class BoxDFactory(BoxFactory):
    dimension_0: float
    dimension_1: float
    dimension_2: float
    in_plane: bool = True

    def create_box(self, particle_factory: ParticleFactory, total_particles: int) -> Box:
        box = Box(
            particles=[particle_factory.create_particle() for _ in range(total_particles)],
            cube = Cube(dimension_0=self.dimension_0, dimension_1=self.dimension_1, dimension_2=self.dimension_2)
            )
        box.force_inside_box(in_plane = self.in_plane)
        return box


@dataclass
class BoxFromDetectorListFactory(BoxFactory):
    detector_list: List[DetectorImage]
    in_plane: bool = True

    def create_box(self, particle_factory: ParticleFactory, total_particles: int) -> Box:
        dimension_0 = np.max([2 * PI / detector.qx_delta for detector in self.detector_list])
        dimension_1 = np.max([2 * PI / detector.qy_delta for detector in self.detector_list])
        dimension_2 = dimension_0
        return BoxDFactory(dimension_0, dimension_1, dimension_2, in_plane=self.in_plane).create_box(particle_factory=particle_factory, total_particles=total_particles)


@dataclass
class BoxListFactory(ABC):

    @abstractmethod
    def create_box_list(self, box_factory: BoxFactory, particle_factory: ParticleFactory) -> List[Box]:
        pass


@dataclass
class BoxListParticleNumberConcentration(BoxListFactory):
    particle_number: int
    nominal_concentration: float

    def create_box_list(self, box_factory: BoxFactory, particle_factory: ParticleFactory) -> List[Box]:
        box_volume = box_factory.create_box(particle_factory, total_particles=0).cube.volume
        total_particle_volume = self.particle_number * particle_factory.calculate_effective_volume()
        particle_conc = total_particle_volume / box_volume
        box_number = int(particle_conc / self.nominal_concentration) + 1
        average_particle_volume = total_particle_volume / self.particle_number
        particle_number_per_box = int(self.nominal_concentration * box_volume / average_particle_volume)
        return [box_factory.create_box(particle_factory, particle_number_per_box) for _ in range(box_number)]
        

@dataclass
class BoxListConcentrationBoxNumber(BoxListFactory):
    box_number: int
    nominal_concentration: float

    def create_box_list(self, box_factory: BoxFactory, particle_factory: ParticleFactory) -> List[Box]:
        box_volume = box_factory.create_box(particle_factory, total_particles=0).cube.volume
        effective_particle_volume = particle_factory.calculate_effective_volume()
        particle_number_per_box = int(self.nominal_concentration / (effective_particle_volume / box_volume))
        return [box_factory.create_box(particle_factory, particle_number_per_box) for _ in range(self.box_number)]


@dataclass
class BoxListParticleNumberBoxNumber(BoxListFactory):
    particle_number: int
    box_number: int

    def create_box_list(self, box_factory: BoxFactory, particle_factory: ParticleFactory) -> List[Box]:
        particle_number_per_box = int(self.particle_number / self.box_number)
        return [box_factory.create_box(particle_factory, particle_number_per_box) for _ in range(self.box_number)]


def gen_from_dict(d: dict, detector_list: List[DetectorImage]) -> BoxFactory:
    box_dimension_0 = d.get("box_dimension_1", 0.0)
    box_dimension_1 = d.get("box_dimension_2", 0.0)
    box_dimension_2 = d.get("box_dimension_3", 0.0)
    in_plane = d.get("in_plane", True)
    if constants.NON_ZERO_LIST([box_dimension_0, box_dimension_1, box_dimension_2]):
        return BoxDFactory(box_dimension_0, box_dimension_1, box_dimension_2, in_plane=in_plane)
    return BoxFromDetectorListFactory(detector_list, in_plane=in_plane)

def gen_list_factory_from_dict(d: dict) -> BoxListFactory:
    nominal_concentration = d.get("nominal_concentration", 0.0)
    particle_number = d.get("particle_number", 0)
    box_number = d.get("box_number", 0)
    
    if particle_number == 0:
        return BoxListConcentrationBoxNumber(box_number=box_number, nominal_concentration=nominal_concentration)
    if box_number == 0:
        return BoxListParticleNumberConcentration(particle_number=particle_number, nominal_concentration=nominal_concentration)
    return BoxListParticleNumberBoxNumber(particle_number=particle_number, box_number=box_number)


    
    

