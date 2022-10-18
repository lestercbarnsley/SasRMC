#%%
from dataclasses import dataclass
from typing import List

import numpy as np

from ..vector import Vector
from ..particles import CoreShellParticle, SphericalParticle
from .particle_factory import ParticleFactory, polydisperse_parameter, rng
from . import command_factory

PI = np.pi

@dataclass
class SphericalCommandFactory(command_factory.CommandFactoryList):
    in_plane: bool = True
    nominal_step_size: float = 100
    nominal_angle_change: float = PI/8
    nominal_rescale_change: float = 0.02
    add_magnetic_commands: bool = True

    def create_command_list(self) -> List[command_factory.CommandFactory]:
        change_by_factor = rng.normal(loc = 1.0, scale = self.nominal_rescale_change)
        position_delta_size = rng.normal(loc = 0.0, scale = self.nominal_step_size)
        position_delta = (Vector.random_vector_xy if self.in_plane else Vector.random_vector)(position_delta_size)
        actual_angle_change = rng.normal(loc = 0.0, scale = self.nominal_angle_change)
        massive_angle_change = rng.uniform(low = -PI, high = PI)

        command_list = [
            command_factory.MoveParticleByFactory(position_delta=position_delta),
            command_factory.JumpToParticleFactory(),
            command_factory.OrbitParticleFactory(actual_angle_change=actual_angle_change),
            command_factory.MoveParticleToFactory(in_plane = self.in_plane),
            command_factory.NuclearMagneticRescaleFactory(change_by_factor=change_by_factor)
            ]
        magnetic_commands = [
            command_factory.RotateMagnetizationFactory(actual_angle_change=actual_angle_change),
            command_factory.RotateMagnetizationFactory(actual_angle_change=massive_angle_change),
            command_factory.FlipMagnetizationFactory()
            ]
        
        return (command_list + magnetic_commands) if self.add_magnetic_commands else command_list


@dataclass
class SphericalParticleFactory(ParticleFactory):
    command_factory: SphericalCommandFactory
    sphere_radius: float
    sphere_polydispersity: float
    sphere_sld: float
    solvent_sld: float
    core_magnetization: float

    def create_particle(self) -> SphericalParticle:
        return SphericalParticle.gen_from_parameters(
            position=Vector.null_vector(),
            magnetization=Vector.random_vector_xy(self.core_magnetization),
            sphere_radius=polydisperse_parameter(loc = self.sphere_radius, polyd=self.sphere_polydispersity),
            sphere_sld=self.sphere_sld,
            solvent_sld=self.solvent_sld
        )

    @classmethod
    def gen_from_dict(cls, d: dict):
        return cls(
            command_factory=SphericalCommandFactory(
                in_plane=d.get("in_plane", True),
                nominal_step_size=d.get("core_radius") / 2,
                add_magnetic_commands=bool(d.get("core_magnetization", 0))
                ),
            sphere_radius=d.get("core_radius"),
            sphere_polydispersity=d.get("core_polydispersity",0.0),
            sphere_sld=d.get("core_sld", 0.0),
            solvent_sld=d.get("solvent_sld", 0.0),
            core_magnetization=d.get("core_magnetization", 0.0)
            )


@dataclass
class CoreShellParticleFactory(ParticleFactory):
    command_factory: SphericalCommandFactory
    core_radius: float
    core_polydispersity: float
    core_sld: float
    shell_thickness: float
    shell_polydispersity: float
    shell_sld: float
    solvent_sld: float
    core_magnetization: float

    def create_particle(self) -> CoreShellParticle:
        return CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=Vector.random_vector_xy(self.core_magnetization),
        core_radius=polydisperse_parameter(loc = self.core_radius, polyd = self.core_polydispersity),
        thickness=polydisperse_parameter(loc = self.shell_thickness, polyd=self.shell_polydispersity),
        core_sld=self.core_sld,
        shell_sld=self.shell_sld,
        solvent_sld=self.solvent_sld)

    @classmethod
    def gen_from_dict(cls, d: dict):
        return cls(
            command_factory=SphericalCommandFactory(
                in_plane=d.get("in_plane", True),
                nominal_step_size=d.get("core_radius") / 2,
                add_magnetic_commands=bool(d.get("core_magnetization", 0))
                ),
            core_radius=d.get("core_radius"),
            sphere_polydispersity=d.get("core_polydispersity",0.0),
            sphere_sld=d.get("core_sld", 0.0),
            solvent_sld=d.get("solvent_sld", 0.0),
            core_magnetization=d.get("core_magnetization", 0.0)
            )




#%%
