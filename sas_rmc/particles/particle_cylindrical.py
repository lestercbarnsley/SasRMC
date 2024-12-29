
from dataclasses import dataclass, field

from typing_extensions import Self
import numpy as np
from scipy import special, integrate
#from scipy.special import jv as j_bessel

from sas_rmc import constants, Vector
from sas_rmc.shapes import Cylinder, Shape
from sas_rmc.particles.particle import Particle

j_bessel = special.jv
PI = constants.PI


@dataclass
class CylindricalParticle(Particle):
    core_cylinder: Cylinder
    cylinder_sld: float
    solvent_sld: float
    magnetization: Vector = field(default_factory=Vector.null_vector)

    def get_position(self) -> Vector:
        return self.core_cylinder.get_position()
    
    def get_orientation(self) -> Vector:
        return self.core_cylinder.get_orientation()
    
    def get_magnetization(self) -> Vector:
        return self.magnetization
    
    def get_volume(self) -> float:
        return self.core_cylinder.get_volume()
    
    def get_shapes(self) -> list[Shape]:
        return [self.core_cylinder]
    
    def is_inside(self, position: Vector) -> bool:
        return self.core_cylinder.is_inside(position)
    
    def get_delta_sld(self) -> float:
        return (self.cylinder_sld - self.solvent_sld) * 1e-6 

    def get_scattering_length(self) -> float:
        return self.get_volume() * self.get_delta_sld()

    def change_position(self, position: Vector) -> Self:
        return type(self)(
            core_cylinder=self.core_cylinder.change_position(position),
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            magnetization=self.magnetization
        )
    
    def change_orientation(self, orientation: Vector) -> Self:
        return type(self)(
            core_cylinder=self.core_cylinder.change_orientation(orientation),
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            magnetization=self.magnetization
        )
    
    def change_magnetization(self, magnetization: Vector) -> Self:
        return type(self)(
            core_cylinder=self.core_cylinder,
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            magnetization=magnetization
        )
    
    def get_loggable_data(self) -> dict:
        loggable_data = super().get_loggable_data()
        return loggable_data | {
            "Core radius" : self.core_cylinder.radius,
            "Core height" : self.core_cylinder.height,
            "Cylinder SLD" : self.cylinder_sld,
            "Solvent SLD" : self.solvent_sld,
            "Total scattering length" : self.get_scattering_length()
        }
    
    def form_profile(self, q_profile: np.ndarray) -> np.ndarray:
        