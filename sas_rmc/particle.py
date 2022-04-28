#%%
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from .vector import Vector, VectorElement, VectorSpace
from .shapes import Shape, Sphere, Cylinder

get_physical_constant = lambda constant_str: constants.physical_constants[constant_str][0]

PI = np.pi
GAMMA_N = np.abs(get_physical_constant('neutron mag. mom. to nuclear magneton ratio')) # This value is unitless
R_0 = get_physical_constant('classical electron radius')
BOHR_MAG = get_physical_constant('Bohr magneton')
B_H_IN_INVERSE_AMP_METRES = (GAMMA_N * R_0 / 2) / BOHR_MAG
MAX_SIZE = 3

sphere_volume = lambda radius: (4 * PI / 3) * radius**3
theta = lambda qR: np.where(qR == 0, 1, 3 * (np.sin(qR) - qR* np.cos(qR)) / (qR**3))
modulus_array = lambda x_arr, y_arr: np.sqrt(x_arr**2 + y_arr**2)


def magnetic_sld_in_angstrom_minus_2(magnetization_vector_in_amp_per_metre: Vector) -> Tuple[float, float, float]:

    # Let us do all calculations in metres, then convert to Angstrom^-2 as the last step
    magnetization = magnetization_vector_in_amp_per_metre
    sld_vector = B_H_IN_INVERSE_AMP_METRES * magnetization / (1e10**2)
    return sld_vector.x, sld_vector.y, sld_vector.z


def round_vector(vector: Vector) -> Tuple[int, int, int]:
    round_vector_comp = lambda comp: int(comp * (2**20))
    return round_vector_comp(vector.x), round_vector_comp(vector.y), round_vector_comp(vector.z)

@dataclass
class FormCache:
    _form_array: dict = field(default_factory=dict)
    _modulated_form_array: dict = field(default_factory=dict)
    _magnetic_form_array: dict = field(default_factory=dict)
    _modulated_magnetic_array: dict = field(default_factory=dict)

    def reset_form(self) -> None: #I'm not sure I need this anymore
        self._form_array = {}
        self._modulated_form_array = {}
        self._magnetic_form_array = {}
        self._modulated_magnetic_array = {}

    @staticmethod
    def access_array(access_dict: dict, arr_tuple: Tuple, calculator_func: Callable[[], object], max_size: int = MAX_SIZE) -> object:
        if arr_tuple not in access_dict:
            if len(access_dict) >= max_size:
                access_dict.pop(list(access_dict.keys())[0])
            form_array_new = calculator_func()
            access_dict[arr_tuple] = form_array_new
        return access_dict[arr_tuple]

    def form_array(self, arr_tuple, form_arr_calculator_func: Callable[[], np.ndarray]):
        return FormCache.access_array(self._form_array, arr_tuple, calculator_func=form_arr_calculator_func)

    def modulated_form_array(self, arr_tuple, modulated_arr_calculator_func: Callable[[], np.ndarray]):
        return FormCache.access_array(self._modulated_form_array, arr_tuple, calculator_func=modulated_arr_calculator_func)

    def magnetic_form_array(self, arr_tuple, magnetic_arr_calculator_func: Callable[[], np.ndarray]):
        return FormCache.access_array(self._magnetic_form_array, arr_tuple, calculator_func=magnetic_arr_calculator_func)

    def modulated_magnetic_array(self, arr_tuple, modulated_magnetic_arr_calculator_func: Callable[[], List[np.ndarray]]):
        return FormCache.access_array(self._modulated_magnetic_array, arr_tuple, calculator_func=modulated_magnetic_arr_calculator_func)


@dataclass
class FormResult:
    form_nuclear: np.ndarray
    form_magnetic_x: np.ndarray
    form_magnetic_y: np.ndarray
    form_magnetic_z: np.ndarray


@dataclass
class Particle(ABC):
    _magnetization: Vector = Vector.null_vector()
    shapes: List[Shape] = field(default_factory = list)
    form_cache: FormCache = field(default_factory= FormCache, repr = False)

    @property
    def volume(self) -> float:
        return np.sum([shape.volume for shape in self.shapes])

    def is_magnetic(self) -> bool:
        return self._magnetization.mag != 0

    @abstractmethod
    def is_spherical(self) -> bool: # Don't allow orientation changes to a spherical particle
        pass

    @property
    @abstractmethod
    def scattering_length(self):
        pass

    @property
    def position(self) -> Vector:
        return self.shapes[0].central_position

    @position.setter
    def position(self, position_new):
        position_delta = position_new - self.position
        for shape in self.shapes:
            shape.central_position = shape.central_position + position_delta

    @property
    def orientation(self) -> Vector:
        return self.shapes[0].orientation

    @orientation.setter
    def orientation(self, orientaton_new):
        if not self.is_spherical(): # Don't allow orientation changes to a spherical particle
            for shape in self.shapes:
                shape.orientation = orientaton_new

    @property
    def magnetization(self) -> Vector:
        return self._magnetization

    @magnetization.setter
    def magnetization(self, magnetization_new: Vector):
        self._magnetization = magnetization_new

    def is_inside(self, position: Vector) -> bool:
        return any(
            [shape.is_inside(position) for shape in self.shapes]
        )
        
    def collision_detected(self, other_particle) -> bool:
        for shape in self.shapes:
            for other_shape in other_particle.shapes:
                if shape.collision_detected(other_shape):
                    return True
        return False

    def random_position_inside(self) -> Vector:
        shape = np.random.choice(self.shapes)
        return shape.random_position_inside()

    @abstractmethod
    def _form_array_calculator(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        pass

    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        form_array_calculator = lambda : self._form_array_calculator(qx_array, qy_array)
        arr_tuple = (id(qx_array), id(qy_array)) + round_vector(self.orientation)
        return self.form_cache.form_array(arr_tuple, form_arr_calculator_func=form_array_calculator)

    def modulated_form_array(self, qx_array, qy_array) -> np.ndarray:
        def modulated_form_array_calc() -> np.ndarray:
            return self.form_array(qx_array, qy_array) * np.exp(1j * (qx_array * self.position.x + qy_array * self.position.y))
        arr_tuple = (id(qx_array), id(qy_array)) + round_vector(self.orientation) + round_vector(self.position)
        return self.form_cache.modulated_form_array(arr_tuple, modulated_arr_calculator_func=modulated_form_array_calc)

    @abstractmethod
    def _magnetic_array_calculator(self, qx_array: np.ndarray, qy_array: np.ndarray) -> np.ndarray:
        pass
    
    def magnetic_form_array(self, qx_array, qy_array) -> np.ndarray:
        magnetic_array_calculator = lambda : self._magnetic_array_calculator(qx_array, qy_array)
        arr_tuple = (id(qx_array), id(qy_array)) + round_vector(self.orientation) + round_vector(self.magnetization)
        return self.form_cache.magnetic_form_array(arr_tuple, magnetic_arr_calculator_func=magnetic_array_calculator)
    
    def magnetic_modulated_array(self, qx_array, qy_array) -> np.ndarray:
        def magnetic_modulated_calc() -> List[np.ndarray]:
            modulated_array = np.exp(1j * (qx_array * self.position.x + qy_array * self.position.y))
            return [f_m * modulated_array for f_m in self.magnetic_form_array(qx_array, qy_array)]
        arr_tuple = (id(qx_array), id(qy_array)) + round_vector(self.orientation) + round_vector(self.magnetization) + round_vector(self.position)
        return self.form_cache.modulated_magnetic_array(arr_tuple, modulated_magnetic_arr_calculator_func=magnetic_modulated_calc)# modulated_arr_calculator_func=magnetic_modulated_calc)

    def form_result(self, qx_array, qy_array) -> FormResult:
        form_nuclear = self.modulated_form_array(qx_array=qx_array, qy_array=qy_array)
        form_magnetic_x, form_magnetic_y, form_magnetic_z = self.magnetic_modulated_array(qx_array=qx_array, qy_array=qy_array)
        return FormResult(
            form_nuclear=form_nuclear,
            form_magnetic_x=form_magnetic_x,
            form_magnetic_y=form_magnetic_y,
            form_magnetic_z=form_magnetic_z,
        )

    def core_repr(self):
        return f'{type(self).__name__}(position = {self.position}, orientation = {self.orientation}, magnetization = {self._magnetization}, volume = {self.volume}, scattering_length = {self.scattering_length})'


@dataclass
class CoreShellParticle(Particle):
    shapes: List[Sphere] = field(
        default_factory=lambda : [Sphere(), Sphere()]
    )
    core_sld: float = 0
    shell_sld: float = 0
    solvent_sld: float = 0
    
    @property
    def volume(self):
        return np.max([shape.volume for shape in self.shapes])

    def is_spherical(self) -> bool:
        return True

    @property
    def scattering_length(self):
        core_sphere = self.shapes[0]
        delta_sld = lambda sld: (sld - self.solvent_sld) * 1e-6
        return delta_sld(self.core_sld - self.shell_sld) * core_sphere.volume + delta_sld(self.shell_sld) * self.volume
 
    @property
    def position(self) -> Vector:
        return super().position

    @position.setter
    def position(self, new_vector: Vector) -> None:
        for shape in self.shapes:
            shape.central_position = new_vector

    def _form_array_calculator(self, qx_array, qy_array):
        core_sphere = self.shapes[0]
        core_shell_sphere = self.shapes[1]
        delta_sld = lambda sld: (sld - self.solvent_sld) * 1e-6
        volume_inner = core_sphere.volume
        volume_outer = self.volume
        q = modulus_array(qx_array, qy_array)
        theta_inner = theta(q * core_sphere.radius)
        theta_outer = theta(q * core_shell_sphere.radius)
        form_amplitude = volume_inner * delta_sld(self.core_sld - self.shell_sld) * theta_inner + volume_outer * delta_sld(self.shell_sld) * theta_outer
        return form_amplitude

    def _magnetic_array_calculator(self, qx_array, qy_array):
        q = modulus_array(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        core_sphere = self.shapes[0]
        volume_inner = core_sphere.volume
        theta_inner = theta(q * core_sphere.radius)
        return [volume_inner * magnetic_sld * theta_inner for magnetic_sld in magnetic_sld_in_angstrom_minus_2(self._magnetization)]

    def collision_detected(self, other_particle) -> bool:
        biggest_shape = self.shapes[np.argmax([shape.volume for shape in self.shapes])]
        for other_shape in other_particle.shapes:
            if biggest_shape.collision_detected(other_shape):
                return True
        return False

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector = None, core_radius = 0, thickness = 0, core_sld = 0, shell_sld = 0, solvent_sld = 0):
        sphere_inner = Sphere(central_position=position, radius=core_radius)
        sphere_outer = Sphere(central_position=position, radius= core_radius + thickness)
        return cls(
            _magnetization=magnetization if magnetization is not None else Vector.null_vector(),
            shapes = [sphere_inner, sphere_outer],
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld
            )
        

def _form_array_numerical(vector_space, qx_array, qy_array, sld_from_vs_fn, xy_axis = 0):
    xy_project = lambda arr: np.average(arr, axis = xy_axis)
    x_arr, y_arr, z_arr = [vector_space.x_arr, vector_space.y_arr, vector_space.z_arr]
    dx_arr, dy_arr, dz_arr = [vector_space.dx_arr, vector_space.dy_arr, vector_space.dz_arr]
    x, y, z = [xy_project(space_arr) for space_arr in [x_arr, y_arr, z_arr]]
    dx, dy, dz= [xy_project(space_arr_partial) for space_arr_partial in [dx_arr, dy_arr, dz_arr]]
    sld = np.sum(sld_from_vs_fn(x_arr, y_arr, z_arr) * dz_arr, axis = xy_axis)
    def form_f(qx, qy):
        return np.sum(sld * np.exp(1j * (qx * x + qy *y)) * dx * dy)
    
    form_function = np.frompyfunc(form_f, 2, 1)
    return np.complex128(form_function(qx_array, qy_array))


@dataclass
class Dumbbell(Particle):
    particle_1: CoreShellParticle = field(default_factory=CoreShellParticle)
    particle_2: CoreShellParticle = field(default_factory=CoreShellParticle)

    @property
    def volume(self) -> float:
        return self.particle_1.volume + self.particle_2.volume

    def is_spherical(self) -> bool:
        return False

    @property
    def scattering_length(self):
        return self.particle_1.scattering_length + self.particle_2.scattering_length

    @property
    def position(self) -> Vector:
        return self.particle_1.position

    @position.setter
    def position(self, new_vector: Vector) -> None:
        delta_position = new_vector - self.position
        self.particle_1.position = self.particle_1.position + delta_position
        self.particle_2.position = self.particle_2.position + delta_position

    @property
    def orientation(self) -> Vector:
        return (self.particle_2.position - self.particle_1.position).unit_vector

    @orientation.setter
    def orientation(self, new_vector: Vector) -> Vector:
        orientation_new = new_vector.unit_vector
        outer_radius = lambda particle : particle.shapes[1].radius
        distance = outer_radius(self.particle_1) + outer_radius(self.particle_2)
        self.particle_2.position = self.particle_1.position + distance * orientation_new

    @property
    def magnetization(self) -> Vector:
        mag_1 = self.particle_1.magnetization
        mag_2 = self.particle_2.magnetization
        return mag_1 if mag_1.mag > mag_2.mag else mag_2

    @property
    def magnetization_minor(self) -> Vector:
        mag_1 = self.particle_1.magnetization
        mag_2 = self.particle_2.magnetization
        return mag_2 if mag_1.mag > mag_2.mag else mag_1

    @magnetization.setter
    def magnetization(self, magnetization_new):
        mag_1 = self.particle_1.magnetization
        mag_2 = self.particle_2.magnetization
        if mag_1.mag > mag_2.mag:
            self.particle_1.magnetization = magnetization_new
        else:
            self.particle_2.magnetization = magnetization_new

    @magnetization_minor.setter
    def magnetization_minor(self, magnetization_new):
        mag_1 = self.particle_1.magnetization
        mag_2 = self.particle_2.magnetization
        if mag_1.mag > mag_2.mag:
            self.particle_2.magnetization = magnetization_new
        else:
            self.particle_1.magnetization = magnetization_new

    def is_inside(self, position: Vector) -> bool:
        return any(
            [particle.is_inside(position) for particle in [self.particle_1, self.particle_2]]
        )

    def collision_detected(self, other_particle) -> bool:
        for particle in [self.particle_1, self.particle_2]:
            if particle.collision_detected(other_particle):
                return True
        return False

    def random_position_inside(self) -> Vector:
        return np.random.choice(
            [particle.random_position_inside() for particle in [self.particle_1, self.particle_2]]
        )

    def _form_array_calculator(self, qx_array, qy_array):
        form_1 = self.particle_1.form_array(qx_array, qy_array)
        form_2 = self.particle_2.form_array(qx_array, qy_array)
        return form_1 + form_2

    def modulated_form_array(self, qx_array, qy_array) -> np.ndarray:
        def modulated_form_array_calcluator():
            form_1 = self.particle_1.modulated_form_array(qx_array, qy_array)
            form_2 = self.particle_2.modulated_form_array(qx_array, qy_array)
            return form_1 + form_2
        arr_tuple = (id(qx_array), id(qy_array)) + round_vector(self.particle_1.position) + round_vector(self.particle_2.position)
        return self.form_cache.modulated_form_array(arr_tuple, modulated_arr_calculator_func=modulated_form_array_calcluator)
        #return form_1 + form_2

    def _magnetic_array_calculator(self, qx_array, qy_array):
        magnetic_1 = self.particle_1.magnetic_form_array(qx_array, qy_array)
        magnetic_2 = self.particle_2.magnetic_form_array(qx_array, qy_array)
        return magnetic_1 + magnetic_2

    def magnetic_modulated_array(self, qx_array, qy_array) -> np.ndarray:
        def modulated_magnetic_array_calcluator():
            magnetic_1 = self.particle_1.magnetic_modulated_array(qx_array, qy_array)
            magnetic_2 = self.particle_2.magnetic_modulated_array(qx_array, qy_array)
            return [m_1 + m_2 for m_1, m_2 in zip(magnetic_1, magnetic_2)]
        arr_tuple = (id(qx_array), id(qy_array)) + round_vector(self.particle_1.magnetization) + round_vector(self.particle_2.magnetization) + round_vector(self.particle_1.position) + round_vector(self.particle_2.position)
        return self.form_cache.modulated_magnetic_array(arr_tuple, modulated_magnetic_arr_calculator_func=modulated_magnetic_array_calcluator)
        
    @classmethod
    def gen_from_parameters(cls, core_radius, seed_radius, shell_thickness, core_sld, seed_sld, shell_sld, solvent_sld, position = Vector.null_vector(), orientation = Vector(0,0,1)):
        core_shell = CoreShellParticle.gen_from_parameters(
            position=position,
            core_radius=core_radius,
            thickness=shell_thickness,
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld
        )
        seed_shell = CoreShellParticle.gen_from_parameters(
            position=position,
            core_radius=seed_radius,
            thickness=shell_thickness,
            core_sld=seed_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld
        )
        dumbell = cls(
            shapes = [core_shell.shapes[1], seed_shell.shapes[1]],
            particle_1 = core_shell,
            particle_2 = seed_shell
            )
        dumbell.orientation = orientation
        return dumbell



@dataclass
class NumericalParticle(Particle):
    '''
    This is an abstract base class for numerical particles
    '''
    vector_space: VectorSpace = None

    @property
    def volume(self):
        return super().volume

    @abstractmethod
    def is_spherical(self) -> bool:
        pass

    @abstractmethod
    def get_sld(self, position: Vector) -> float:
        pass

    def get_scattering_length(self, element: VectorElement) -> float:
        position = element.position
        sld = self.get_sld(position)
        return sld * element.volume

    def sld_from_vector_space(self, vector_space: VectorSpace = None) -> np.ndarray:
        '''def get_scattering_length(element: VectorElement) -> float:
            position = element.position
            sld = self.get_sld(position)
            return sld * element.volume'''
        vector_space_elements = self.vector_space if vector_space is None else vector_space
        return vector_space_elements.array_from_elements(lambda element : self.get_scattering_length(element))

    @property
    @abstractmethod
    def scattering_length(self):
        return np.sum(self.sld_from_vector_space())
       
    def _form_array_calculator(self, qx_array, qy_array):
        xy_axis = 2
        flat_sld = np.sum(self.sld_from_vector_space(), axis=xy_axis)
        x = np.average(self.vector_space.x, axis = xy_axis)
        y = np.average(self.vector_space.y, axis = xy_axis)
        def form_f(qx, qy):
            return np.sum(
                flat_sld * np.exp(1j * (qx * x + qy * y))
                )
        form_calculator = np.frompyfunc(form_f, 2, 1)
        return form_calculator(qx_array, qy_array).astype(np.complex128)

    def show_particle(self):
        xy_axis = 2
        sld = self.sld_from_vector_space()
        flat_sld = np.sum(sld, axis=xy_axis)
        plt.imshow(flat_sld)
        plt.show()

    def _magnetic_array_calculator(self, qx_array, qy_array):
        q = modulus_array(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        is_inside_magnet = lambda x, y, z: int(self.shape_outer.is_inside_shape(Vector(x, y, z)))
        is_inside_fn = np.frompyfunc(is_inside_magnet, 3, 1)
        theta = _form_array_numerical(self.vector_space, qx_array=qx_array, qy_array=qy_array, sld_from_vs_fn=is_inside_fn)
        #return [b_H * self.volume * theta for b_H in b_H_vec_from_moment(self._magnetization, self.volume)]
        return [self.volume * magnetic_sld * theta for magnetic_sld in magnetic_sld_in_angstrom_minus_2(self._magnetization)]

    
@dataclass
class SphereNumerical(NumericalParticle):
    shapes: List[Sphere] = field(
        default_factory=lambda : [Sphere()]
    )
    sphere_sld: float = 0
    solvent_sld: float = 0

    @property
    def volume(self):
        return super().volume

    def is_spherical(self) -> bool:
        return True

    @property
    def scattering_length(self):
        return (self.sphere_sld - self.solvent_sld) * self.volume

    def get_sld(self, position: Vector) -> float:
        relative_position = position# - self.position
        return (self.sphere_sld - self.solvent_sld) * 1e-6 if self.is_inside(relative_position) else 0

    @classmethod
    def gen_from_parameters(cls, radius, sphere_sld, solvent_sld, pixel_size = None, position = Vector.null_vector(), max_size = None, vector_space = None):
        dx = pixel_size if pixel_size is not None else radius / 10
        vs = vector_space if vector_space is not None else VectorSpace.gen_from_bounds(-radius, +radius, int(2 * radius / dx), -radius, +radius, int(2 * radius / dx), -radius, +radius, int(2 * radius / dx))
        return cls(
            shapes=[Sphere(central_position = position, radius = radius)],
            vector_space = vs,
            sphere_sld = sphere_sld,
            solvent_sld = solvent_sld
            )


@dataclass
class CylinderNumerical(NumericalParticle):
    shapes: List[Cylinder] = field(
        default_factory=lambda : [Cylinder()]
    )
    cylinder_sld: float = 0
    solvent_sld: float = 0

    @property
    def volume(self):
        return super().volume

    def is_spherical(self) -> bool:
        return False

    @property
    def scattering_length(self):
        return (self.cylinder_sld - self.solvent_sld) * self.volume

    def get_sld(self, position: Vector) -> float:
        relative_position = position - self.position
        return (self.cylinder_sld - self.solvent_sld) * 1e-6 if self.is_inside(relative_position) else 0

    @classmethod
    def gen_from_parameters(cls, radius, height, cylinder_sld, solvent_sld, pixel_size = None, position = Vector.null_vector(), orientation = Vector(0,0,1), max_size = None, vector_space = None):
        dx = pixel_size if pixel_size is not None else radius / 10
        dimension = np.max([2*radius, height])
        vs = vector_space if vector_space is not None else VectorSpace.gen_from_bounds(-dimension, +dimension, int(2 * radius / dx), -dimension, +dimension, int(2 * radius / dx), -radius, +radius, int(2 * radius / dx))
        return cls(
            shapes = [Cylinder(
                central_position = position,
                orientation = orientation,
                radius = radius,
                height = height
                )],
            vector_space= vs,
            cylinder_sld=cylinder_sld,
            solvent_sld=solvent_sld
            )


@dataclass
class NumericalParticleCustom(NumericalParticle):
    get_sld_function: Callable[[Vector], float] = None

    def is_spherical(self) -> bool:
        return False

    def get_sld(self, position: Vector) -> float:
        return self.get_sld_function(position)

    def scattering_length(self):
        return super().scattering_length



if __name__ == "__main__":
    pass


# %%
