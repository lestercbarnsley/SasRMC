#%%

from abc import ABC, abstractmethod

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass
import numpy as np

from sas_rmc import Vector, constants
from sas_rmc.factories import parse_data
from sas_rmc.factories.evaluator_factory import ProfileType, infer_profile_type
from sas_rmc.particles import ParticleResult
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticleForm, CoreShellParticleProfile
from sas_rmc.particles.particle_cylindrical import CylindricalParticle, CylindricalParticleForm

rng = constants.RNG


def polydisperse_parameter(loc: float, polyd: float) -> float:
    return rng.normal(loc = loc, scale = loc * polyd)

def rotate_xy_vector(vector: Vector, angle: float) -> Vector:
    x = vector.x * np.cos(angle) - vector.y * np.sin(angle)
    y = vector.x * np.sin(angle) + vector.y * np.cos(angle)
    return Vector(x, y, z = vector.z)

def polydisperse_orientation(starting_vector: Vector, polyd: float) -> Vector:
    angle = rng.normal(loc = 0, scale=polyd)
    return rotate_xy_vector(starting_vector, angle)
    

@pydantic_dataclass
class ParticleFactory(ABC):

    @abstractmethod
    def create_particle_result(self) -> ParticleResult:
        pass


@pydantic_dataclass
class CoreShellParticleFactory(ParticleFactory):
    profile_type: ProfileType
    core_magnetization: float
    core_radius: float
    core_polydispersity: float
    shell_thickness: float
    shell_polydispersity: float
    core_sld: float
    shell_sld: float
    solvent_sld: float

    def create_particle_form(self) -> CoreShellParticleForm:
        return CoreShellParticleForm.gen_from_parameters(
            position=Vector.null_vector(),
            magnetization=self.core_magnetization * Vector.random_vector_xy(),
            core_radius=polydisperse_parameter(self.core_radius, self.core_polydispersity),
            thickness=polydisperse_parameter(self.shell_thickness, self.shell_polydispersity),
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )
    
    def create_particle_profile(self) -> CoreShellParticleProfile:
        return CoreShellParticleProfile.gen_from_parameters(
            position=Vector.null_vector(),
            magnetization=self.core_magnetization * Vector.random_vector_xy(),
            core_radius=polydisperse_parameter(self.core_radius, self.core_polydispersity),
            thickness=polydisperse_parameter(self.shell_thickness, self.shell_polydispersity),
            core_sld=self.core_sld,
            shell_sld=self.shell_sld,
            solvent_sld=self.solvent_sld
        )
    
    def create_particle_result(self) -> ParticleResult:
        match self.profile_type:
            case ProfileType.DETECTOR_IMAGE:
                return self.create_particle_form()
            case ProfileType.PROFILE:
                return self.create_particle_profile()
    
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame], profile_type: ProfileType):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(profile_type = profile_type, **value_frame)
    

@pydantic_dataclass
class CylindricalParticleFactory(ParticleFactory):
    profile_type: ProfileType
    core_magnetization: float
    cylinder_radius: float
    radius_polydispersity: float
    cylinder_height: float
    height_polydisperity: float
    orientation_polydispersity: float
    cylinder_sld: float
    solvent_sld: float

    def create_particle_result(self) -> ParticleResult:
        orientation = polydisperse_orientation(Vector(0, 1), self.orientation_polydispersity)
        return CylindricalParticleForm.gen_from_parameters(
            radius=polydisperse_parameter(self.cylinder_radius, self.radius_polydispersity),
            height=polydisperse_parameter(self.cylinder_height, self.height_polydisperity),
            orientation=orientation,
            cylinder_sld=self.cylinder_sld,
            solvent_sld=self.solvent_sld,
            magnetization=Vector.random_vector_xy(self.core_magnetization)
        )
    
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame], profile_type: ProfileType):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(profile_type = profile_type, **value_frame)


def create_particle_factory_from(dataframes: dict[str, pd.DataFrame]) -> ParticleFactory:
    value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
    particle_type = value_frame['particle_type']
    profile_type = infer_profile_type(dataframes)
    match particle_type:
        case "CoreShellParticle":
            return CoreShellParticleFactory.create_from_dataframes(dataframes, profile_type)
        case "CylindricalParticle":
            return CylindricalParticleFactory.create_from_dataframes(dataframes, profile_type)
        case _:
            raise NotImplementedError("Other particle types currently aren't implemented")
    



#%%
