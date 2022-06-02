#%%
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from .array_cache import method_array_cache, round_vector
from .vector import Vector, VectorElement, VectorSpace#, dot
from .shapes import Shape, Sphere, Cylinder, collision_detected

get_physical_constant = lambda constant_str: constants.physical_constants[constant_str][0]

PI = np.pi
GAMMA_N = np.abs(get_physical_constant('neutron mag. mom. to nuclear magneton ratio')) # This value is unitless
R_0 = get_physical_constant('classical electron radius')
BOHR_MAG = get_physical_constant('Bohr magneton')
B_H_IN_INVERSE_AMP_METRES = (GAMMA_N * R_0 / 2) / BOHR_MAG

sphere_volume = lambda radius: (4 * PI / 3) * radius**3
theta = lambda qR: np.where(qR == 0, 1, 3 * (np.sin(qR) - qR* np.cos(qR)) / (qR**3))
modulus_array = lambda x_arr, y_arr: np.sqrt(x_arr**2 + y_arr**2)


def magnetic_sld_in_angstrom_minus_2(magnetization_vector_in_amp_per_metre: Vector) -> Tuple[float, float, float]:
    # Let us do all calculations in metres, then convert to Angstrom^-2 as the last step
    magnetization = magnetization_vector_in_amp_per_metre
    sld_vector = B_H_IN_INVERSE_AMP_METRES * magnetization / (1e10**2)
    return sld_vector.x, sld_vector.y, sld_vector.z


@dataclass
class FormResult:
    form_nuclear: np.ndarray
    form_magnetic_x: np.ndarray
    form_magnetic_y: np.ndarray
    form_magnetic_z: np.ndarray


@dataclass
class Particle(ABC):
    """Abstract base class for Particle type.

    When you make your own particle type, you should inherit from this class. This is an abstract class, so it cannot be instantiated itself, but it shows a template of how a Particle class should be written, and provides abstract methods that should have implementations written by the user.

    Attributes
    ----------
    shapes : List[Shape]
        List of shapes used to calculate collision detection.
    solvent_sld : float, optional
        The SLD of the solvent phase in 1E-6 * Angstrom^-2.
    _magnetization : Vector, optional
        A vector describing the magnetization of the magnetic phase. Units are Amps/metre.

    """
    _magnetization: Vector = field(default_factory = Vector.null_vector)
    shapes: List[Shape] = field(default_factory = list)
    solvent_sld: float = 0

    @property
    def volume(self) -> float:
        """The volume of the particle.

        Returns
        -------
        float
            The volume of the particle in units Angstrom^3
        """
        return np.sum([shape.volume for shape in self.shapes])

    def is_magnetic(self) -> bool:
        """Is the particle magnetic?

        Returns
        -------
        bool
            Returns True if the particle is magnetic.
        """
        return self._magnetization.mag != 0

    def delta_sld(self, sld: float) -> float:
        """The contrast between an SLD value and the SLD of the solvent phase

        Note the unit change between the input and return values.

        Parameters
        ----------
        sld : float
            An SLD value to test against, in 1E-6 * Angstrom^-2.

        Returns
        -------
        float
            The contrast between the SLD value and solvent SLD in Angstrom^-2
        """
        return (sld - self.solvent_sld) * 1e-6

    @property
    @abstractmethod
    def scattering_length(self) -> float:
        """The total scattering length of the particle in Angstroms.

        This is an abstract method. A user will need to write their own implementation of this method when inheriting from the Particle class.

        Returns
        -------
        float
            The total scattering length of the particle in Angstroms
        """
        pass

    @property
    def position(self) -> Vector:
        """A vector to represent the position of the particle.

        Usually this is the centre of the particle. This may also be the centre of one of the Shape objects in self.shapes.

        Returns
        -------
        Vector
            The position of the particle in Angstroms.
        """
        return self.shapes[0].central_position

    def set_position(self, position: Vector) -> None:
        """Set the position vector of a Particle object.

        Parameters
        ----------
        position : Vector
            The new position vector of the particle, in Angstrom.
        """
        position_delta = position - self.position
        for shape in self.shapes:
            shape.central_position = shape.central_position + position_delta

    def is_spherical(self) -> bool: # Don't allow orientation changes to a spherical particle
        """Check if the particle is spherical.

        In this context, a particle is spherical if an orientation change has no effect on the particle. Therefore, a particle that consists of a group of spheres may not be spherical itself, but it would be if all these particles share the same centre position. This is essentially how the default implementation works, but you are STRONGLY encouraged to write your own implementation when you make a new Particle type (even though this isn't an abstract method). After all, you yourself know if your particle is spherical or not. You do know if your particle is spherical... right?

        Returns
        -------
        bool
            Returns True if the particle is spherical
        """
        same_location = lambda : all(round_vector(shape.central_position) == round_vector(self.shapes[0].central_position) for shape in self.shapes)
        return all(isinstance(shape, Sphere) for shape in self.shapes) and same_location()

    @property
    def orientation(self) -> Vector:
        return self.shapes[0].orientation

    def set_orientation(self, orientation: Vector) -> None:
        if not self.is_spherical(): # Don't allow orientation changes to a spherical particle
            for shape in self.shapes:
                shape.orientation = orientation

    @property
    def magnetization(self) -> Vector:
        return self._magnetization

    def set_magnetization(self, magnetization: Vector) -> None:
        self._magnetization = magnetization

    def is_inside(self, position: Vector) -> bool:
        """Method for determining if a position in space is inside a particle.

        Parameters
        ----------
        position : Vector
            Position vector to test.

        Returns
        -------
        bool
            Returns True if the position is inside the particle.
        """
        return any(shape.is_inside(position) for shape in self.shapes) # Use a generator here to take advantage of lazy iteration
        
        
    def collision_detected(self, other_particle) -> bool:
        return collision_detected(self.shapes, other_particle.shapes)

    def random_position_inside(self) -> Vector:
        shape = np.random.choice(self.shapes)
        return shape.random_position_inside()

    @abstractmethod
    def _form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        pass

    @method_array_cache
    def form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        return self._form_array(qx_array, qy_array, orientation)
        
    @method_array_cache
    def modulated_form_array(self, qx_array, qy_array, position: Vector, orientation: Vector) -> np.ndarray:
        return self.form_array(qx_array, qy_array, orientation) * np.exp(1j * (position * (qx_array, qy_array)))#(qx_array * position.x + qy_array * position.y))
        
    @abstractmethod
    def _magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    @method_array_cache
    def magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._magnetic_form_array(qx_array, qy_array, orientation, magnetization)
        
    @method_array_cache
    def magnetic_modulated_array(self, qx_array, qy_array, position: Vector, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        modulated_array = np.exp(1j * (position * (qx_array, qy_array)))#(qx_array * position.x + qy_array * position.y))
        return [fm * modulated_array for fm in self.magnetic_form_array(qx_array, qy_array, orientation, magnetization)]
        
    def form_result(self, qx_array, qy_array) -> FormResult:
        form_nuclear = self.modulated_form_array(qx_array=qx_array, qy_array=qy_array, position=self.position, orientation=self.orientation)
        form_magnetic_x, form_magnetic_y, form_magnetic_z = self.magnetic_modulated_array(qx_array=qx_array, qy_array=qy_array, position=self.position, orientation=self.orientation, magnetization=self.magnetization)
        return FormResult(
            form_nuclear=form_nuclear,
            form_magnetic_x=form_magnetic_x,
            form_magnetic_y=form_magnetic_y,
            form_magnetic_z=form_magnetic_z,
        )

    def get_loggable_data(self) -> dict:
        return {
        'Particle type': type(self).__name__,
        **self.position.to_dict("Position"),
        **self.orientation.to_dict("Orientation"),
        **self.magnetization.to_dict("Magnetization"),
        'Volume' : self.volume,
        'Total scattering length' : self.scattering_length,
    }


def form_array_sphere(radius: float, sld: float, q_array: np.ndarray) -> np.ndarray:
    volume = sphere_volume(radius)
    theta_arr = theta(q_array * radius)
    return sld * volume * theta_arr


@dataclass
class CoreShellParticle(Particle):
    shapes: List[Sphere] = field(
        default_factory=lambda : [Sphere(), Sphere()]
    )
    core_sld: float = 0
    shell_sld: float = 0
    
    @property
    def volume(self):
        return np.max([shape.volume for shape in self.shapes])

    def is_spherical(self) -> bool:
        return True

    @property
    def scattering_length(self):
        core_sphere = self.shapes[0]
        return self.delta_sld(self.core_sld - self.shell_sld) * core_sphere.volume + self.delta_sld(self.shell_sld) * self.volume
 
    @property
    def position(self) -> Vector:
        return super().position

    def set_position(self, position: Vector) -> None:
        for shape in self.shapes:
            shape.central_position = position

    def set_orientation(self, orientation: Vector) -> None:
        super().set_orientation(self.orientation) # This will block changes to orientation

    def _form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        core_sphere = self.shapes[0]
        core_shell_sphere = self.shapes[1]
        q = modulus_array(qx_array, qy_array)
        core_form = form_array_sphere(
            radius = core_sphere.radius, 
            sld = self.delta_sld(self.core_sld - self.shell_sld), 
            q_array=q)
        shell_form = form_array_sphere(
            radius = core_shell_sphere.radius, 
            sld = self.delta_sld(self.shell_sld), 
            q_array = q)
        return core_form + shell_form

    def _magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = modulus_array(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        core_sphere = self.shapes[0]
        return [form_array_sphere(core_sphere.radius, magnetic_sld, q) for magnetic_sld in magnetic_sld_in_angstrom_minus_2(magnetization)]

    def collision_detected(self, other_particle: Particle) -> bool:
        biggest_shape = self.shapes[np.argmax([shape.volume for shape in self.shapes])]
        return collision_detected([biggest_shape], other_particle.shapes)

    def get_loggable_data(self) -> dict:
        core_radius = self.shapes[0].radius
        overall_radius = self.shapes[1].radius
        thickness = overall_radius - core_radius
        data = {
            'Particle type': "",
            'Core radius': core_radius,
            'Shell thickness': thickness,
            'Core SLD': self.core_sld,
            'Shell SLD' : self.shell_sld,
            'Solvent SLD': self.solvent_sld,
        }
        data.update(super().get_loggable_data())
        return data

    @classmethod
    def gen_from_parameters(cls, position: Vector, magnetization: Vector = None, core_radius: float = 0, thickness: float = 0, core_sld: float = 0, shell_sld: float = 0, solvent_sld: float = 0):
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
    def form_f(qx: float, qy: float) -> float:
        return np.sum(sld * np.exp(1j * (Vector(qx, qy) * (x, y))) * dx * dy)
    
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

    def set_position(self, position: Vector) -> None:
        orientation = self.orientation
        self.particle_1.set_position(position)
        self.set_orientation(orientation)

    @property
    def orientation(self) -> Vector:
        return (self.particle_2.position - self.particle_1.position).unit_vector

    def set_orientation(self, orientation: Vector) -> None:
        orientation_new = orientation.unit_vector
        outer_radius = lambda particle : particle.shapes[1].radius
        distance = outer_radius(self.particle_1) + outer_radius(self.particle_2)
        self.particle_2.set_position(self.particle_1.position + distance * orientation_new)

    @property
    def magnetization(self) -> Vector:
        mag_1 = self.particle_1.magnetization
        mag_2 = self.particle_2.magnetization
        return mag_1 if mag_1.mag > mag_2.mag else mag_2

    def set_magnetization(self, magnetization: Vector) -> None:
        mag_1 = self.particle_1.magnetization
        mag_2 = self.particle_2.magnetization
        if mag_1.mag > mag_2.mag:
            self.particle_1.set_magnetization(magnetization)
        else:
            self.particle_2.set_magnetization(magnetization)

    def is_inside(self, position: Vector) -> bool:
        return any(particle.is_inside(position) for particle in [self.particle_1, self.particle_2])

    def collision_detected(self, other_particle: Particle) -> bool:
        sphere_particles = [self.particle_1, self.particle_2]
        return any(sphere_particle.collision_detected(other_particle) for sphere_particle in sphere_particles)

    def random_position_inside(self) -> Vector:
        random_particle = np.random.choice([self.particle_1, self.particle_2])
        return random_particle.random_position_inside()
        '''return np.random.choice(
            [particle.random_position_inside() for particle in [self.particle_1, self.particle_2]]
        )'''

    def _form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        # This will never actually be called because we override modulated form array
        form_1 = self.particle_1.form_array(qx_array, qy_array, orientation=self.particle_1.orientation)
        form_2 = self.particle_2.form_array(qx_array, qy_array, orientation=self.particle_2.orientation)
        return form_1 + form_2

    @method_array_cache
    def modulated_form_array(self, qx_array, qy_array, position: Vector, orientation: Vector) -> np.ndarray:
        form_1 = self.particle_1.modulated_form_array(qx_array, qy_array, position=self.particle_1.position, orientation=self.particle_1.orientation)
        form_2 = self.particle_2.modulated_form_array(qx_array, qy_array, position=self.particle_2.position, orientation=self.particle_2.orientation)
        return form_1 + form_2

    def _magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # This will never actually be called because we override modulated form array
        magnetic_1 = self.particle_1.magnetic_form_array(qx_array, qy_array, orientation=self.particle_1.orientation, magnetization=self.particle_1.magnetization)
        magnetic_2 = self.particle_2.magnetic_form_array(qx_array, qy_array, orientation=self.particle_2.orientation, magnetization=self.particle_2.magnetization)
        return [m1 + m2 for m1, m2 in zip(magnetic_1, magnetic_2)]

    @method_array_cache
    def magnetic_modulated_array(self, qx_array, qy_array, position: Vector, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        magnetic_1 = self.particle_1.magnetic_modulated_array(qx_array, qy_array, position=self.particle_1.position, orientation=self.particle_1.orientation, magnetization=self.particle_1.magnetization)
        magnetic_2 = self.particle_2.magnetic_modulated_array(qx_array, qy_array, position=self.particle_2.position, orientation=self.particle_2.orientation, magnetization=self.particle_2.magnetization)
        return [m1 + m2 for m1, m2 in zip(magnetic_1, magnetic_2)]

    def get_loggable_data(self) -> dict:
        core_radius = self.particle_1.shapes[0].radius
        overall_radius = self.particle_1.shapes[1].radius
        thickness = overall_radius - core_radius
        data = {
            'Particle type' : "",
            'Core radius': core_radius,
            'Seed radius' : self.particle_2.shapes[0].radius,
            'Shell thickness': thickness,
            'Core SLD': self.particle_1.core_sld,
            'Seed SLD': self.particle_2.core_sld,
            'Shell SLD' : self.particle_1.shell_sld,
            'Solvent SLD': self.particle_1.solvent_sld,
        }
        data.update(super().get_loggable_data())
        data.update(self.particle_1.magnetization.to_dict('MagnetizationCore'))
        data.update(self.particle_2.magnetization.to_dict('MagnetizationSeed'))
        
        return data

    @classmethod
    def gen_from_parameters(cls, core_radius, seed_radius, shell_thickness, core_sld, seed_sld, shell_sld, solvent_sld, position: Vector = None, orientation: Vector = None, core_magnetization: Vector = None, seed_magnetization: Vector = None):
        particle_position = position if position else Vector.null_vector()
        particle_orientation = orientation if orientation else Vector(0, 1, 0)
        core_shell = CoreShellParticle.gen_from_parameters(
            position=particle_position,
            core_radius=core_radius,
            thickness=shell_thickness,
            core_sld=core_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            magnetization=core_magnetization if core_magnetization else Vector.null_vector()
        )
        seed_shell = CoreShellParticle.gen_from_parameters(
            position=particle_position,
            core_radius=seed_radius,
            thickness=shell_thickness,
            core_sld=seed_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            magnetization=seed_magnetization if seed_magnetization else Vector.null_vector()
        )
        dumbell = cls(
            shapes = [core_shell.shapes[1], seed_shell.shapes[1]],
            particle_1 = core_shell,
            particle_2 = seed_shell
            )
        dumbell.set_orientation(particle_orientation)
        return dumbell


def numerical_form_array(flat_sld: np.ndarray, x_arr: np.ndarray, y_arr: np.ndarray, qx: float, qy: float) -> float:
    return np.sum(flat_sld * np.exp(1j * (Vector(qx, qy) * (x_arr, y_arr))))


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
    def get_sld(self, position: Vector) -> float:
        pass

    def get_scattering_length(self, element: VectorElement) -> float:
        position = element.position
        sld = self.get_sld(position)
        return sld * element.volume

    def sld_from_vector_space(self, vector_space: VectorSpace = None) -> np.ndarray:
        vector_space_elements = self.vector_space if vector_space is None else vector_space
        return vector_space_elements.array_from_elements(lambda element : self.get_scattering_length(element))

    @property
    @abstractmethod
    def scattering_length(self):
        return np.sum(self.sld_from_vector_space())

    def _form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector) -> np.ndarray:
        xy_axis = 2
        flat_sld = np.sum(self.sld_from_vector_space(), axis=xy_axis)
        x = np.average(self.vector_space.x, axis = xy_axis)
        y = np.average(self.vector_space.y, axis = xy_axis)
        form_f = lambda qx, qy: numerical_form_array(flat_sld, x, y, qx, qy)
        form_calculator = np.frompyfunc(form_f, 2, 1)
        return form_calculator(qx_array, qy_array).astype(np.complex128)

    def show_particle(self):
        xy_axis = 2
        sld = self.sld_from_vector_space()
        flat_sld = np.sum(sld, axis=xy_axis)
        plt.imshow(flat_sld)
        plt.show()

    def _magnetic_form_array(self, qx_array: np.ndarray, qy_array: np.ndarray, orientation: Vector, magnetization: Vector) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # I need to think about how to implement this properly
        q = modulus_array(qx_array, qy_array)
        if not self.is_magnetic():
            return [np.zeros(q.shape) for _ in range(3)]
        is_inside_magnet = lambda x, y, z: int(self.shape_outer.is_inside_shape(Vector(x, y, z)))
        is_inside_fn = np.frompyfunc(is_inside_magnet, 3, 1)
        theta = _form_array_numerical(self.vector_space, qx_array=qx_array, qy_array=qy_array, sld_from_vs_fn=is_inside_fn)
        #return [b_H * self.volume * theta for b_H in b_H_vec_from_moment(self._magnetization, self.volume)]
        return [self.volume * magnetic_sld * theta for magnetic_sld in magnetic_sld_in_angstrom_minus_2(magnetization)]

    
@dataclass
class SphereNumerical(NumericalParticle):
    shapes: List[Sphere] = field(
        default_factory=lambda : [Sphere()]
    )
    sphere_sld: float = 0

    @property
    def volume(self):
        return super().volume

    def is_spherical(self) -> bool:
        return True

    @property
    def scattering_length(self):
        return self.delta_sld(self.sphere_sld) * self.volume

    def get_sld(self, position: Vector) -> float:
        relative_position = position - self.position
        return self.delta_sld(self.sphere_sld) if self.is_inside(relative_position) else self.delta_sld(self.solvent_sld)

    def get_loggable_data(self) -> dict:
        data = {
            'Particle type' : "",
            'Sphere radius': self.shapes[0].radius,
            'Sphere SLD' : self.sphere_sld
            }
        data.update(super().get_loggable_data())
        return data

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

    @property
    def volume(self):
        return super().volume

    def is_spherical(self) -> bool:
        return False

    @property
    def scattering_length(self):
        return self.delta_sld(self.cylinder_sld) * self.volume

    def get_sld(self, position: Vector) -> float:
        relative_position = position - self.position
        return self.delta_sld(self.cylinder_sld) if self.is_inside(relative_position) else self.delta_sld(self.solvent_sld)

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
class NumericalDumbell(NumericalParticle):
    shapes: List[Shape] = field(
        default_factory=lambda : [Sphere(), Sphere(), Sphere(), Sphere()]
    )
    core_sld: float = 0
    seed_sld: float = 0
    shell_sld: float = 0
    _centre_to_centre: float = 0

    @property
    def centre_to_centre(self) -> float:
        return self._centre_to_centre

    def set_centre_to_centre(self, centre_to_centre: float) -> None:
        self._centre_to_centre = centre_to_centre

    @property
    def position(self) -> Vector:
        return self.shapes[0].central_position

    def set_position(self, position: Vector) -> None:
        orientation = self.orientation
        self.shapes[0].central_position = position
        self.shapes[1].central_position = position
        self.set_orientation(orientation)

    @property
    def orientation(self) -> Vector:
        return (self.shapes[2].central_position - self.shapes[0].central_position).unit_vector

    def set_orientation(self, orientation: Vector) -> None:
        orientation_new = orientation.unit_vector
        particle_2_position = self.position + (self.centre_to_centre * orientation_new)
        self.shapes[2].central_position = particle_2_position
        self.shapes[3].central_position = particle_2_position

    def get_sld(self, position: Vector) -> float:
        if self.shapes[2].is_inside(position):
            return self.delta_sld(self.seed_sld)
        if self.shapes[0].is_inside(position):
            return self.delta_sld(self.core_sld)
        if self.shapes[1].is_inside(position) or self.shapes[3].is_inside(position):
            return self.delta_sld(self.shell_sld)
        return self.delta_sld(self.solvent_sld)

    @classmethod
    def gen_from_parameters(cls, core_radius: float, seed_radius: float, shell_thickness: float, centre_to_centre: float, core_sld: float, seed_sld: float, shell_sld: float, solvent_sld: float, position: Vector = None, orientation: Vector = None):
        position_ = position if position else Vector.null_vector()
        orientation_ = orientation if orientation else Vector(0,1,0)
        numerical_dumbell = cls(
            shapes=[
                Sphere(central_position=position_,radius=core_radius),
                Sphere(central_position=position_, radius = core_radius + shell_thickness),
                Sphere(central_position=position_, radius = seed_radius),
                Sphere(central_position=position_, radius=seed_radius + shell_thickness)
                ],
            core_sld=core_sld,
            seed_sld=seed_sld,
            shell_sld=shell_sld,
            solvent_sld=solvent_sld,
            _centre_to_centre = centre_to_centre,
        )
        numerical_dumbell.set_orientation(orientation_)
        return numerical_dumbell


@dataclass
class NumericalParticleCustom(NumericalParticle):
    '''
    This is highly experimental, please don't use this unless you can handle the consequences
    '''
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
