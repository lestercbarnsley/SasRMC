
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.special import jv as j_bessel
from scipy.integrate import quad, fixed_quad

from .particle import Particle
from ..vector import Vector, broadcast_to_numpy_array
from ..shapes.shapes import Shape, Cylinder
from .. import constants

#def long_cylinder_average_fraction(cylinder_radius: float, radius: float) -> float:

PI = constants.PI


@dataclass
class CylindricalParticle(Particle):
    _shapes: List[Cylinder] = field(
        default_factory=lambda : [Cylinder()]
    )
    cylinder_sld: float = 0

    @property
    def shapes(self) -> List[Cylinder]:
        return super().shapes

    def _change_shapes(self, shape_list: List[Shape]):
        return CylindricalParticle(
            _magnetization = self._magnetization,
            _shapes = shape_list,
            solvent_sld=self.solvent_sld,
            cylinder_sld=self.cylinder_sld
        )

    def set_magnetization(self, magnetization: Vector):
        return CylindricalParticle(
            _magnetization = magnetization,
            _shapes = self._shapes,
            solvent_sld=self.solvent_sld,
            cylinder_sld=self.cylinder_sld
        )

    @property
    def volume(self):
        return super().volume

    def is_spherical(self) -> bool:
        return False

    @property
    def scattering_length(self) -> float:
        return self.delta_sld(self.cylinder_sld) * self.volume

    def get_sld(self, position: Vector) -> float:
        relative_position = position - self.position
        return self.delta_sld(self.cylinder_sld) if self.is_inside(relative_position) else self.delta_sld(self.solvent_sld)

    def get_average_sld(self, radius: float) -> float:
        cylinder_radius = self.shapes[0].radius
        cylinder_height = self.shapes[0].height
        if 2 * cylinder_radius < cylinder_height:
            if radius < cylinder_radius:
                return self.cylinder_sld
            if radius > cylinder_height / 2:
                return self.solvent_sld
            average_fraction = (radius - np.sqrt(radius**2 - cylinder_radius**2)) / radius
            return self.solvent_sld + (self.cylinder_sld - self.solvent_sld) * average_fraction
        else:
            if radius < cylinder_height / 2:
                return self.cylinder_sld
            if radius > cylinder_radius:
                return self.solvent_sld
            average_fraction = cylinder_height / (2 * radius)
            return self.solvent_sld + (self.cylinder_sld - self.solvent_sld) * average_fraction

    @classmethod
    def gen_from_parameters(cls, radius, height, cylinder_sld, solvent_sld = 0, position = None, orientation = None):
        if position is None:
            position = Vector.null_vector()
        if orientation is None:
            orientation = Vector(0,1,0)
        return cls(
            _shapes = [Cylinder(
                central_position = position,
                orientation = orientation,
                radius = radius,
                height = height
                )],
            cylinder_sld=cylinder_sld,
            solvent_sld=solvent_sld
            )

    def form_cylinder(self, q: float, alpha: float) -> float:
        yL = q * self.shapes[0].height * np.cos(alpha) / 2
        yR = q * self.shapes[0].radius * np.sin(alpha)
        scale =  2 * (self.cylinder_sld - self.solvent_sld) * self.volume
        return scale * np.sinc(yL / PI) * np.where(yR == 0, 1/2, j_bessel(1,  yR) / yR)

    def rotation(self, q):
        y, *_ = fixed_quad(lambda alpha : self.form_cylinder(q, alpha)**2, 0, PI/2 , n = 10)
        #y, *_ = quad(lambda alpha : self.form_cylinder(q, alpha)**2, 0, PI/2 )
        return y

    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        if constants.NON_ZERO_LIST(qy_array):
            pass
        else:
            return np.sqrt(broadcast_to_numpy_array(qx_array, lambda q : self.rotation(q)))
            


    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().magnetic_form_array(qx_array, qy_array, orientation, magnetization)

