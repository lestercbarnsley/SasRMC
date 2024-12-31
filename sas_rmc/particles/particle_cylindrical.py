#%%

from dataclasses import dataclass, field

from typing_extensions import Self
import numpy as np
from scipy import special

from sas_rmc import Vector, vector
from sas_rmc.shapes import Cylinder, Shape
from sas_rmc.particles.particle import Particle


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
    
    def get_sld_at_position(self, relative_position: Vector) -> float:
        position = relative_position + self.get_position()
        if self.core_cylinder.is_inside(position):
            return (self.cylinder_sld - self.solvent_sld) * 1e-6
        return self.solvent_sld * 1e-6
    
    def sld_sum_along_line(self, x: float, y: float, num: int = 101) -> float:
        extent = self.core_cylinder.height
        z_line = np.linspace(-extent, +extent, num = num)
        line = [Vector(x, y, z) for z in z_line]
        sld = np.array([self.get_sld_at_position(relative_position) for relative_position in line])
        return np.sum(sld * np.gradient(z_line))
    
    def form_arr(self, q: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        h_arg = q * self.core_cylinder.height * np.cos(alpha)
        r_arg = q * self.core_cylinder.radius * np.sin(alpha)
        return 2 * (self.cylinder_sld - self.solvent_sld) * self.get_volume() * special.spherical_jn(0, h_arg) * special.j0(r_arg)#np.where(r_arg != 0, special.j0(r_arg) / r_arg, 1)
    
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        
        q = np.sqrt(qx_array**2 + qy_array**2)
        alpha = np.acos(vector.dot((qx_array / q, qy_array / q), self.get_orientation().unit_vector.to_tuple()))
        return self.form_arr(q, alpha)
    


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    cylinder = CylindricalParticle(
        core_cylinder=Cylinder(10, 300, Vector.null_vector(), Vector(0, 1, 0)),
        cylinder_sld=6.9,
        solvent_sld=0
    )
    qx, qy = np.meshgrid(
        np.linspace(-0.4, 0.4, num = 101),
        np.linspace(-0.4, 0.4, num = 101)
    )
    f = cylinder.form_array(qx, qy)


    plt.imshow(np.log(np.real((f * f.conj()).astype(np.complex64))))
    plt.show()

    print(np.acos(1))
        


#%%
