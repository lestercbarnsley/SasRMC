#%%

from abc import ABC, abstractmethod

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc import Vector, constants
from sas_rmc.factories import parse_data
from sas_rmc.factories.evaluator_factory import ProfileType, infer_profile_type
from sas_rmc.particles import ParticleResult
from sas_rmc.particles.particle_core_shell_spherical import CoreShellParticleForm, CoreShellParticleProfile
from sas_rmc.factories.parse_data import coerce_types

rng = constants.RNG


def polydisperse_parameter(loc: float, polyd: float) -> float:
    return rng.normal(loc = loc, scale = loc * polyd)

@coerce_types
def create_core_shell_particle(
                core_radius: float,
                core_polydispersity: float,
                core_sld: float,
                shell_thickness: float,
                shell_polydispersity: float,
                shell_sld: float,
                solvent_sld: float,
                core_magnetization: float,
                ) -> CoreShellParticleForm:
    return CoreShellParticleForm.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=core_magnetization * Vector.random_vector_xy(),
        core_radius=polydisperse_parameter(core_radius, core_polydispersity),
        thickness=polydisperse_parameter(shell_thickness, shell_polydispersity),
        core_sld=core_sld,
        shell_sld=shell_sld,
        solvent_sld=solvent_sld
    )
    

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
        if self.profile_type == ProfileType.DETECTOR_IMAGE:
            return self.create_particle_form()
        return self.create_particle_profile()
    
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame], profile_type: ProfileType):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(profile_type = profile_type, **value_frame)

def create_from_dataframes(dataframes: dict[str, pd.DataFrame]) -> ParticleFactory:
    value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
    particle_type = value_frame['particle_type']
    profile_type = infer_profile_type(dataframes)
    if particle_type == "CoreShellParticle":
        return CoreShellParticleFactory.create_from_dataframes(dataframes, profile_type)
    raise NotImplementedError("Other particle types currently aren't implemented")
    



#%%
