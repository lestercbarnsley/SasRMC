#%%
from typing import Callable

import numpy as np

import sas_rmc
from sas_rmc.array_cache import array_cache
from sas_rmc.form_calculator import box_intensity_average
from sas_rmc.vector import Vector, broadcast_to_numpy_array
from sas_rmc.particles import CoreShellParticle, CylindricalParticle
from sas_rmc.shapes.shapes import Cube
from sas_rmc.result_calculator import AnalyticalCalculator, NumericalProfileCalculator, ProfileCalculator
from sas_rmc.box_simulation import Box
from sas_rmc.profile_calculator import box_profile_calculator

from matplotlib import pyplot as plt

@array_cache
def functional_cache(arr: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def adder(arr_2: np.ndarray):
        #print(id(arr))
        return arr + arr_2
    return adder

add_to_threes = functional_cache(np.array([3,3,3,3]))

for _ in range(10):
    add_to_threes(np.random.rand(4))

def numerical_test():
    profile_calculator = ProfileCalculator(q_array=np.linspace(1e-3, .5, num = 1000), r_array=np.linspace(0,130000, num = 10000))
    particle_number = 100
    particles = [CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(),
        core_radius=100,
        thickness= 10,
        core_sld=6,
        shell_sld=0
    ) for _ in range(particle_number)]
    particles = [CylindricalParticle.gen_from_parameters(
        radius= 103,
        height=1000,
        cylinder_sld=6
    ) for _ in range(particle_number)]
    box = Box(particles=particles, cube = Cube(dimension_0=100000, dimension_1=3000, dimension_2=10000))
    box.force_inside_box(in_plane=True)

    intensity = box_profile_calculator(box, profile_calculator)
    plt.loglog(profile_calculator.q_array, intensity)
    plt.show()

    box.plot_particle_positions('b.')

def average_sld_func_for_cylinder_test():
    r_array=np.linspace(0,1300, num = 200)
    radius = 20
    height = 1000
    particle = sas_rmc.particle.CylindricalParticle.gen_from_parameters(radius = radius, height=height, cylinder_sld=6, solvent_sld=0)
    sld_averager = lambda r : np.average([particle.get_sld(Vector.random_vector(r)) for _ in range(1000)])
    average_sld = broadcast_to_numpy_array(r_array, sld_averager)
    plt.plot(r_array, average_sld)
    plt.plot(r_array, np.where(r_array > radius, 6e-6 * ( (radius / r_array)**3+ 0*radius**3 / (6 * r_array**3)), 6e-6))
    
    plt.show()

    plt.loglog(r_array, average_sld)
    plt.loglog(r_array, np.where(r_array > radius, 6e-6 * ( (radius / r_array)**3 + 0* radius**3 / (6 * r_array**3)), 6e-6))
    
   
    plt.show()

    plt.plot(r_array, average_sld)
    plt.plot(r_array, np.where(r_array > radius, 6e-6 * ((r_array - np.sqrt(r_array**2 - radius**2))/r_array), 6e-6))
    plt.show()

    plt.loglog(r_array, average_sld)
    plt.loglog(r_array, np.where(r_array > radius, 6e-6 * ((r_array - np.sqrt(r_array**2 - radius**2))/r_array), 6e-6))
    plt.show()

    sld_average = [1e-6 * particle.get_average_sld(r) for r in r_array]
    for plotter in [plt.plot, plt.loglog]:
        plotter(r_array, average_sld)
        plotter(r_array, sld_average)
        plt.show()

def core_shell_test():
    core_shell = CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(), core_radius=50.6, core_sld=6.975, thickness=15.1, shell_sld = 0.0776, solvent_sld=5.646
    )
    q = np.linspace(3e-3, 0.3, num=1000)
    intensity = box_intensity_average([Box([core_shell], cube = Cube(dimension_0=6000, dimension_1=6000, dimension_2=6000))],AnalyticalCalculator(q, 0*q))
    file_maker = sas_rmc.simulator_factory.generate_file_path_maker(r"J:\Uni\Programming\SasRMC\data\results", "core_shell")
    plt.loglog(q, intensity + 1e-6)
    np.savetxt(file_maker("", "txt"), [(q_i, i_i) for q_i, i_i in zip(q, intensity)], delimiter='\t')
    plt.show()

if __name__ == "__main__":
    numerical_test()
    #core_shell_test()
    #testing_average_sld_func_for_cylinder()
    
#%%


    

#%%

