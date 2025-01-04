#%%

from dataclasses import dataclass, field

from typing_extensions import Self
import numpy as np
from scipy import special

from sas_rmc import Vector, vector
from sas_rmc.array_cache import array_cache
from sas_rmc.particles import FormResult
from sas_rmc.particles.particle_form import ParticleArray
from sas_rmc.shapes import Cylinder, Shape
from sas_rmc.particles.particle import Particle
from sas_rmc.form_calculator import q_squared


@array_cache
def q_magnitude(qx_array: np.ndarray, qy_array: np.ndarray, offset: float = 1e-16) -> np.ndarray:
    return np.sqrt(q_squared(qx_array, qy_array, offset=offset))

@array_cache
def calculate_alpha_angle(qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
    q = q_magnitude(qx_array, qy_array)
    return np.acos(vector.dot((qx_array / q, qy_array / q), orientation.unit_vector.to_tuple()))


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
        raise NotImplementedError()
    
    def form_array_with_orientation(self, q: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        h_arg = q * self.core_cylinder.height * np.cos(alpha)
        r_arg = q * self.core_cylinder.radius * np.sin(alpha)
        return 2 * (self.cylinder_sld - self.solvent_sld) * self.get_volume() * special.spherical_jn(0, h_arg) * np.where(r_arg == 0, 1/2, special.j1(r_arg) / r_arg)
    
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        q = q_magnitude(qx_array, qy_array)
        alpha = calculate_alpha_angle(qx_array, qy_array, self.get_orientation())
        return self.form_array_with_orientation(q, alpha)
    
    @classmethod
    def gen_from_parameters(cls, radius: float, height: float, orientation: Vector, cylinder_sld: float, solvent_sld: float, magnetization: Vector | None = None):
        return cls(
            core_cylinder=Cylinder(
                radius=radius,
                height=height,
                central_position=Vector.null_vector(),
                orientation=orientation,
            ),
            cylinder_sld=cylinder_sld,
            solvent_sld=solvent_sld,
            magnetization=magnetization if magnetization is not None else Vector.null_vector()
        )
    

@dataclass
class CylindricalParticleForm(ParticleArray):
    bound_particle: CylindricalParticle

    def get_bound_particle(self) -> Particle:
        return self.bound_particle
    
    def change_particle(self, particle: Particle) -> Self:
        if not isinstance(particle, CylindricalParticle):
            raise TypeError()
        return type(self)(bound_particle = particle)
    
    def form_result(self, qx_array: np.ndarray, qy_array: np.ndarray) -> FormResult:
        # Temperarily ignore magnetic scattering
        return FormResult(
            form_nuclear=self.bound_particle.form_array(qx_array, qy_array),
            form_magnetic_x= 0 * qx_array,
            form_magnetic_y=0 * qx_array,
            form_magnetic_z=0 * qx_array
        )
    
    @classmethod
    def gen_from_parameters(cls, radius: float, height: float, orientation: Vector, cylinder_sld: float, solvent_sld: float, magnetization: Vector | None = None):
        bound_particle = CylindricalParticle.gen_from_parameters(
            radius=radius,
            height=height,
            orientation=orientation,
            cylinder_sld=cylinder_sld,
            solvent_sld=solvent_sld,
            magnetization=magnetization
        )
        return cls(bound_particle = bound_particle)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    cylinder = CylindricalParticle(
        core_cylinder=Cylinder(300, 300, 
            Vector.null_vector(),
            Vector(0, 0, 1)),
        cylinder_sld=6.9,
        solvent_sld=0
    )
    qx, qy = np.meshgrid(
        np.linspace(-0.04, 0.04, num = 101),
        np.linspace(-0.04, 0.04, num = 101)
    )
    f = cylinder.form_array(qx, qy)


    plt.imshow(np.log(np.real((f * f.conj()).astype(np.complex64))))
    plt.show()

    plt.semilogy(np.real(f * f.conj()))
    plt.show()
    

#%%
