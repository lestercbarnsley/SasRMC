#%%
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize
import pandas as pd
from uncertainties import unumpy, ufloat

import sas_rmc

from sas_rmc import DetectorImage, Vector, VectorSpace, SimulatedDetectorImage, DetectorConfig, Polarization
from sas_rmc import result_calculator
from sas_rmc.box_simulation import Box
from sas_rmc.command_writer import BoxWriter
from sas_rmc.form_calculator import box_intensity_average
from sas_rmc.particle import CoreShellParticle, Dumbbell, Particle
from sas_rmc.result_calculator import AnalyticalCalculator, NumericalCalculator, ParticleNumerical
from sas_rmc.scattering_simulation import SimulationParam, SimulationParams
from sas_rmc.shapes import Cube
from sas_rmc.simulator_factory import is_float,  subtract_buffer_intensity
from sas_rmc.fitter import smear_simulated_intensity

file_maker = sas_rmc.simulator_factory.generate_file_path_maker(Path.cwd(), description="figures")
file_maker = sas_rmc.simulator_factory.generate_file_path_maker(r'J:\Uni\Programming\SasRMC\data\results', description="figures")


PI = np.pi
mod = lambda arr: np.real(arr * np.conj(arr))

def default_detector_image() -> SimulatedDetectorImage:
    FILE = r'J:\Uni\Sorted results\User Experiments\p13209 - Feygenson\data\ASCII-I\I-pandas-unpolarized-SM-16586-S9DH-B--2200mT-KWS1.DAT'
    DETECTOR_CONFIG_C14D14 = DetectorConfig(
        detector_distance_in_m=14,
        collimation_distance_in_m=14,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*6e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        wavelength_spread=0.1,
        polarization=Polarization.UNPOLARIZED
        )
    detector_image = SimulatedDetectorImage.gen_from_txt(FILE, DETECTOR_CONFIG_C14D14)
    return detector_image

def box_from_detector(detector_list: DetectorImage, particle_list: List[Particle]) -> Box:
    dimension_0 = np.max([2 * PI / detector.qx_delta for detector in detector_list])
    dimension_1 = np.max([2 * PI / detector.qy_delta for detector in detector_list])
    dimension_2 = dimension_0
    return Box(
        particles=particle_list,
        cube = Cube(
            dimension_0=dimension_0,
            dimension_1=dimension_1,
            dimension_2=dimension_2
        )
    )


def show_particle(particle: ParticleNumerical, numerical_calculator: NumericalCalculator):
    xy_axis = 2
    #sld = particle.sld_from_vector_space()
    vector_space = numerical_calculator.vector_space
    get_sld = lambda position : particle.get_sld(position - particle.position)
    sld = numerical_calculator.sld_from_vector_space(get_sld, particle.delta_sld)
    flat_sld = np.sum(sld, axis=xy_axis)
    x_arr = np.average(vector_space.x, axis= xy_axis)
    y_arr = np.average(vector_space.y, axis = xy_axis)
    
    return x_arr, y_arr, flat_sld


def adjust_i_array(i_array, i, len_i):
    if i <= 0:
        return [0,1,2]
    if i >= len_i -2:
        return [len_i -3, len_i -2, len_i -1]
    return i_array

def subset_array(arr, target, out_len):
    vals = sorted([int((i + 1) / 2) * (+1 if i % 2 else -1) for i in range(out_len)])
    nominal_arg = np.argmin(np.abs(arr - target))
    args = [nominal_arg + val for val in vals]
    if any([arg < 0 for arg in args]):
        return [i for i in range(out_len)]
    if any([arg >= len(arr) for arg in args]):
        return sorted([len(arr) - i - 1 for i in range(out_len)])
    return args
    



def intensity_at_qxqy(intensity_array, qx_array, qy_array, qx_i, qy_i):
    if not np.min(qx_array) < qx_i < np.max(qx_array):
        return 0
    if not np.min(qy_array) < qy_i < np.max(qy_array):
        return 0
    i_s = subset_array(qx_array[0:,], qx_i, 5)
    j_s = subset_array(qy_array[:,0], qy_i, 5)
    
    qx = [qx_array[0,ii] for ii in i_s]
    qy = [qy_array[jj,0] for jj in j_s]
    i_sub = [[intensity_array[jj, ii] for jj in j_s] for ii in i_s]
    interpolator = interpolate.interp2d(qx, qy, i_sub, copy = False)
    return np.average(interpolator(qx_i, qy_i))
    

def radial_average(i_array: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, sector: Tuple[float, float] = (0, np.inf), num_of_angles = 180):
    q_mod = np.sqrt(qx_array**2 + qy_array**2)

    
    q = np.arange(
        start = np.min(np.where(i_array != 0, q_mod, np.inf)),
        stop = np.max(q_mod),
        step = np.average(np.diff(qx_array[0,:]))
        )
    
    def i_of_q(q_c):
        angles = np.linspace(-PI, +PI, num = num_of_angles)
        sector_filter = np.abs(angles - sector[0]) < sector[1] / 2
        qx, qy = q_c * np.cos(angles)[sector_filter], q_c * np.sin(angles)[sector_filter]
        
        if len(qx) == 0:
            return 0
        
        intensities = np.frompyfunc(lambda x, y: intensity_at_qxqy(i_array, qx_array, qy_array, x, y),2,1)(qx, qy).astype(np.float64)
        
        if len(intensities[intensities != 0]) == 0:
            return 0
        return np.average(intensities[intensities > 0])

    averager = np.frompyfunc(i_of_q, 1, 1)
    return q, averager(q).astype(np.float64)

def sin_2_theta(theta, a, offset = 0, theta_offset = 0 ):
    return a * (np.sin(theta + theta_offset)**2) + offset
    

def unpol_analysis(detector_up: DetectorImage, detector_down: DetectorImage, intensity_getter = None):
    qx_array = detector_up.qX
    qy_array = detector_up.qY
    q_mod = np.sqrt(qx_array**2 + qy_array**2)

    '''step = np.average(np.diff(qx_array[0,:]))
    print(step)'''
    
    q = np.arange(
        start = np.min(np.where(detector_up.intensity != 0, q_mod, np.inf)),
        stop = np.max(q_mod),
        step = np.average(np.diff(qx_array[0,:]))
        )
    
    getter = (lambda d: d.intensity) if intensity_getter is None else intensity_getter
    intensity_up = getter(detector_up)
    intensity_down = getter(detector_down)
    i_diff_array = intensity_up - intensity_down
    i_unpol_array = (intensity_up + intensity_down) /2

    def fn_fm(q_c):
        angles = np.linspace(-PI, +PI, num = 180)
        qx, qy = q_c * np.cos(angles + PI/2), q_c * np.sin(angles + PI/2)
        if len(qx) == 0:
            return 0
        fnfm_intensities = np.frompyfunc(lambda x, y: intensity_at_qxqy(i_diff_array, qx_array, qy_array, x, y),2,1)(qx, qy).astype(np.float64)
        if len(fnfm_intensities[fnfm_intensities!=0]) < 6:
            return 0, 0, 0
        popt, pcov = optimize.curve_fit(
            sin_2_theta, angles[fnfm_intensities!=0], fnfm_intensities[fnfm_intensities!=0], p0 = [np.max(fnfm_intensities)]#, 0, 0]
        )
        fnfm = ufloat(popt[0] / 4, pcov[0,0] / 4)

        unpol_intensities = np.frompyfunc(lambda x, y: intensity_at_qxqy(i_unpol_array, qx_array, qy_array, x, y),2,1)(qx, qy).astype(np.float64)

        popt_u, pcov_u = optimize.curve_fit(
            sin_2_theta, angles[unpol_intensities!=0], unpol_intensities[unpol_intensities!=0], p0 = [np.max(unpol_intensities), np.min(unpol_intensities)]#, 0]
        )
        fn2 = ufloat(popt_u[1] , pcov_u[1,1])

        fm2 = fnfm**2 / fn2
        #fm2 = #ufloat(popt_u[0], pcov_u[0,0])
        return fn2, fnfm, fm2


    sin_2_fitter = np.frompyfunc(fn_fm, 1, 3)
    f_nuc_uf, cross_term_uf, f_mag_uf = sin_2_fitter(q)
    nom_and_stdev = lambda arr : (unumpy.nominal_values(arr), unumpy.std_devs(arr))
    f_nuc, f_nuc_err = nom_and_stdev(f_nuc_uf)
    cross_term, cross_term_err = nom_and_stdev(cross_term_uf)
    f_mag, f_mag_err = nom_and_stdev(f_mag_uf)
    as_type_and_filter = lambda arr : arr.astype(np.float64)[f_nuc != 0]
    return q[f_nuc != 0], as_type_and_filter(f_nuc), as_type_and_filter(cross_term), as_type_and_filter(f_mag), as_type_and_filter(f_nuc_err), as_type_and_filter(cross_term_err), as_type_and_filter(f_mag_err)
    

FONT_SIZE = 14

def guinier_model(q, i_0, r_g):
    i =  i_0 * np.exp(-(q * r_g)**2 / 3)
    return i


def figure_form_factors():
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches((4 * 3,3.5 * 2))

    

    '''def onion_sld(position: Vector) -> float:
        if position.mag < 100:
            return 10
        if position.mag < 120:
            return 5
        return 0'''

    core_shell_particle = CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=Vector.null_vector(),
        core_radius=100,
        thickness=20,
        core_sld=10,
        shell_sld=5,
        solvent_sld=0,
    )

    '''def dumbbell_sld(position: Vector) -> float:
        shell_thickness = 10
        position_1 = Vector.null_vector()
        position_2 = 100 * (Vector(1, 1).unit_vector)#Vector(70, 70, 30)
        radius_1 = 90
        radius_2 = 80
        if (position - position_1).mag < radius_1:
            return 10
        if (position - position_2).mag < radius_2:
            return 20
        if (position - position_1).mag < (radius_1 + shell_thickness):
            return 5
        if (position - position_2).mag < (radius_2 + shell_thickness):
            return 5
        return 0'''
    dumbbell = Dumbbell.gen_from_parameters(
        core_radius = 80,
        seed_radius=90,
        shell_thickness=10,
        core_sld = 20,
        seed_sld = 10,
        shell_sld = 5,
        solvent_sld=0,
        centre_to_centre_distance=100,
        orientation=Vector(-1,-1).unit_vector,
        position=100 * (Vector(1,1).unit_vector)
    )

    '''onion_particle = NumericalParticleCustom(
        get_sld_function=onion_sld,
        vector_space=VectorSpace.gen_from_bounds(
            x_min = -160, x_max= 160, x_num = 40,
            y_min = -160, y_max= 160, y_num = 40,
            z_min = -160, z_max= 160, z_num = 40,
        )
    )'''
    detector = default_detector_image()
    box = box_from_detector([detector], [core_shell_particle])
    qx, qy = detector.qX, detector.qY
    numerical_calculator = NumericalCalculator(qx_array=qx, qy_array=qy, vector_space=VectorSpace.gen_from_bounds(
            x_min = -160, x_max= 160, x_num = 40,
            y_min = -160, y_max= 160, y_num = 40,
            z_min = -160, z_max= 160, z_num = 40,
        ))
    x_arr, y_arr, flat_sld = show_particle(core_shell_particle, numerical_calculator=numerical_calculator)

    
    
    
    axs[0,0].pcolormesh(x_arr / 10, y_arr / 10, flat_sld, shading = 'auto')
    #axs[0,0].set_xlabel('X (nm)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[0,0].set_ylabel('Y (nm)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[0,0].xaxis.set_ticklabels([])
    #axs[0,0].text(-14, +13, '(a) Core-shell particle', color = "white")
    axs[0,0].text(0.05, 0.95, '(a)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,0].transAxes)
    axs[0,0].text(0.05, 0.05, 'Core-shell particle', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,0].transAxes)
    
    

    #form_array = core_shell_particle.form_array(qx, qy, orientation=Vector(0,0,1))
    form_array = numerical_calculator.form_result(core_shell_particle).form_nuclear
    intensity = box_intensity_average([box], numerical_calculator)
    axs[0,1].contourf(qx, qy, intensity, levels = 20)
    #axs[0,1].set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[0,1].set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[0,1].xaxis.set_ticklabels([])
    axs[0,1].text(0.05, 0.95, '(b)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,1].transAxes)
    
    qx_large, qy_large = np.meshgrid(np.linspace(-0.5, +0.5, 400),np.linspace(-0.5, +0.5, 400))
    numerical_calculator_large = NumericalCalculator(qx_array=qx_large, qy_array=qy_large, vector_space=VectorSpace.gen_from_bounds(
            x_min = -160, x_max= 160, x_num = 40,
            y_min = -160, y_max= 160, y_num = 40,
            z_min = -160, z_max= 160, z_num = 40,
        ))
    #form_array_2 = core_shell_particle.form_array(qx_large, qy_large, orientation=Vector(0,0,1))
    form_array_2 = numerical_calculator_large.form_result(core_shell_particle).form_nuclear
    i_array = box_intensity_average([box], numerical_calculator_large)#mod(form_array_2)
    #q = np.linspace(2e-3, 4e-1, num  = 300)
    
    
    q, i_1d = radial_average(i_array, qx_large, qy_large)#, q)
    q = q[i_1d != 0]
    i_1d = i_1d[i_1d != 0]
    i_1d = i_1d[q < 4e-1]
    q = q[q < 4e-1]

    analytical_calculator = AnalyticalCalculator(q, 0 * q)

    #i_core_shell = mod(core_shell_particle.form_array(q, np.zeros(q.shape), orientation=Vector.null_vector()))
    i_core_shell = box_intensity_average([box], analytical_calculator)

    axs[0,2].loglog(q, i_core_shell , 'r-', label = "Analytical")
    axs[0,2].loglog(q, i_1d , 'b.', label = "Numerical")
    #axs[0,2].set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    axs[0,2].set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)
    axs[0,2].xaxis.set_ticklabels([])
    bottom, top = axs[0,2].get_ylim()
    axs[0,2].set_ylim(bottom, top * 4)
    axs[0,2].text(0.05, 0.95, '(c)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,2].transAxes)
    axs[0,2].legend(loc = 'upper right')

    #axs[0,0].imshow()

    '''dumbell_particle = NumericalParticleCustom(
        get_sld_function=dumbbell_sld,
        vector_space=VectorSpace.gen_from_bounds(
            x_min = -160, x_max= 160, x_num = 40,
            y_min = -160, y_max= 160, y_num = 40,
            z_min = -160, z_max= 160, z_num = 40,
        )
    )'''
    dumbell_particle = dumbbell
    x_arr, y_arr, flat_sld = show_particle(dumbell_particle, numerical_calculator=numerical_calculator)
    axs[1,0].pcolormesh(x_arr / 10, y_arr / 10, flat_sld, shading = 'auto')
    axs[1,0].set_xlabel('X (nm)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[1,0].set_ylabel('Y (nm)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[1,0].text(0.05, 0.95, '(d)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,0].transAxes)
    axs[1,0].text(0.05, 0.05, 'Dumbbell particle', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,0].transAxes)
    
    #form_array = dumbell_particle.form_array(qx, qy, orientation=Vector(0,0,1))
    box = box_from_detector([detector], [dumbell_particle])
    
    #form_array = numerical_calculator.form_result(dumbell_particle).form_nuclear
    intensity = box_intensity_average([box], numerical_calculator)
    
    axs[1,1].contourf(qx, qy, intensity, levels = 20)
    axs[1,1].set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[1,1].set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[1,1].text(0.05, 0.95, '(e)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,1].transAxes)
    
    #qx_large, qy_large = np.meshgrid(np.linspace(-0.5, +0.5, 200),np.linspace(-0.5, +0.5, 200))
    #form_array_2 = dumbell_particle.form_array(qx_large, qy_large, orientation=Vector(0,0,1))
    form_array_2 = numerical_calculator_large.form_result(dumbell_particle).form_nuclear
    
    i_array = box_intensity_average([box], numerical_calculator_large)

    

    q = np.linspace(2e-3, 4e-1, num  = 300)
    
    
    q, i_1d = radial_average(i_array, qx_large, qy_large)#, q)
    q = q[i_1d != 0]
    i_1d = i_1d[i_1d != 0]
    i_1d = i_1d[q < 4e-1]
    q = q[q < 4e-1]

    popt, pcov = optimize.curve_fit(guinier_model, q, i_1d, p0= [1, 100], method='lm')
    print(popt, pcov)

    fitted_i_0, fitted_rg = popt

    axs[1,2].loglog(q[q * fitted_rg < (1.3 * 3)], guinier_model(q, fitted_i_0, fitted_rg)[q * fitted_rg < (1.3 * 3)], 'r-', label = "Guinier")
    axs[1,2].loglog(q, i_1d , 'b.', label = "Numerical")
    axs[1,2].set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    axs[1,2].set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)
    bottom, top = axs[1,2].get_ylim()
    axs[1,2].set_ylim(bottom, top * 4)
    axs[1,2].text(0.05, 0.95, '(f)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,2].transAxes)
    axs[1,2].legend(loc = 'upper right')

    for ax_ in axs:
        for ax in ax_:
            ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(file_maker("form_factors", ".eps"))
    fig.savefig(file_maker("form_factors", ".pdf"))
    

def intensity_matrix_from_detector(detector: SimulatedDetectorImage):
    intensity_matrix = np.where(detector.qX < 0, detector.experimental_intensity, detector.simulated_intensity * detector.shadow_factor)
    return np.log(intensity_matrix)

def detector_image(ax, qx, qy, intensity, title: str, letter: str, levels = None):
    levels = levels if levels is not None else np.linspace(np.min(intensity[~np.isinf(intensity)]), np.max(intensity[~np.isinf(intensity)]), num=30)
    ax.contourf(qx, qy, intensity, levels = levels, cmap = 'jet')
    ax.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    ax.text(0.025, 0.05, "Experiment",fontsize =  FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.975, 0.05, "Simulation",fontsize =  FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.975, 0.95, title, color = "black", fontsize = FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.05, 0.95, letter, color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    
def particle_positions(ax, positions, letter, x_lims, y_lims):
    ax.plot(positions[:,0] /10, positions[:,1] /10, 'b.') 
    ax.set_ylabel(r'Y (nm)',fontsize =  FONT_SIZE)
    ax.text(0.05, 0.95, letter, color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.set_xlim(np.min(x_lims), np.max(x_lims))
    ax.set_ylim(np.min(y_lims), np.max(y_lims))

def share_lims(axs, ax_getter, ax_setter):
    ax_lims = np.array([ax_getter(ax) for ax in axs])
    new_ax = np.min(ax_lims[:,0]), np.max(ax_lims[:,1])
    for ax in axs:
        ax_setter(ax, new_ax)

def profiles(ax, experimental_average_fn, simulated_average_fn, sector_tuples, letter: str, factor_difference = 100):
    for i, sector_tuple in enumerate(sector_tuples):
        factor = factor_difference**i
        angle, col = sector_tuple
        q, sector_average = experimental_average_fn(angle)
        ax.loglog(
            q[sector_average > 0], factor * sector_average[sector_average > 0], col + '.'
            )
        q_sim, sector_average_sim = simulated_average_fn(angle)
        ax.loglog(
            q_sim[sector_average_sim > 0], factor * sector_average_sim[sector_average_sim > 0], col + '-'
        )
    ax.set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)
    ax.text(0.05, 0.95, letter, color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

def box_to_position_array(box: Box) -> np.ndarray:
    return np.array([[particle.position.x, particle.position.y] for particle in box.particles])

def figure_particle_maps():
    #from matplotlib.gridspec import GridSpec

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

    
    fig.set_size_inches((4.2 * 3,3.5 * 3))

    unsmeared_positions = np.genfromtxt(
        fname = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Unsmeared_particle_positions.txt",
        skip_header = 1
    )
    
    detector = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Unsmeared_detector.txt",
        skip_header=1,
    )
    
    intensity_matrix = intensity_matrix_from_detector(detector)

    smeared_positions = np.genfromtxt(
        fname = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Smeared_particle_positions_200.txt",
        skip_header = 1
    )
    smeared_detector = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Smeared_detector_200.txt",
        skip_header=1,
    )
    
    smeared_intensity_matrix = intensity_matrix_from_detector(smeared_detector)

    saxs_detectors, saxs_boxes, _ = read_simulation_output(r"J:\Uni\Programming\SasRMC\data\results\20220818211718_this_is_a_test.xlsx")
    saxs_detector = saxs_detectors[0]

    def of_two_arrays(a, b, func):
        def clean_and_process_data(arr):
            a_1 = arr[~np.isnan(arr)]
            return func(a_1[~np.isinf(a_1)])
        return func([clean_and_process_data(a), clean_and_process_data(b)])

    print(np.min(intensity_matrix[~np.isnan(intensity_matrix)]))

    levels = np.linspace(
        start = of_two_arrays(intensity_matrix, smeared_intensity_matrix, np.min),
        stop = of_two_arrays(intensity_matrix, smeared_intensity_matrix, np.max),
        num  = 30
    )
    print(levels)

    detector_image(ax1, detector.qX, detector.qY, intensity_matrix, title = "SANS no smearing", letter = "(a)", levels = levels)
    detector_image(ax4, smeared_detector.qX, smeared_detector.qY, smeared_intensity_matrix, title = "SANS with smearing", letter = "(d)", levels = levels)
    detector_image(ax7, saxs_detector.qX, saxs_detector.qY, intensity_matrix_from_detector(saxs_detector),title = "SAXS no smearing", letter = "(g)")
    
    x_lims = (-14070.66526562795/20, +14070.66526562795/20)
    y_lims = (-14070.66526562795/20, +14070.66526562795/20)

    particle_positions(ax2, unsmeared_positions, letter = "(b)", x_lims=x_lims, y_lims=y_lims)
    particle_positions(ax5, smeared_positions, letter = "(e)", x_lims=x_lims, y_lims=y_lims)
    particle_positions(ax8, box_to_position_array(saxs_boxes[0]), letter = "(h)", x_lims=x_lims, y_lims=y_lims)


    sector_angle = 10 * PI/180
    num_of_angles = 360

    experimental_average_maker = lambda detector : (lambda angle : radial_average(
        detector.experimental_intensity * detector.shadow_factor,
        detector.qX,
        detector.qY,
        sector = (angle, sector_angle ),
        num_of_angles=num_of_angles)
        )
    simulatated_average_maker = lambda detector : (lambda angle : radial_average(
        detector.simulated_intensity * detector.shadow_factor,
        detector.qX,
        detector.qY,
        sector = (angle, sector_angle ),
        num_of_angles=num_of_angles)
        )

    sector_tuples = [(0, 'b'), ( PI/6, 'r'), (2*PI/6,'k' ), (3*PI/6,'g' ) ]

    profiles(ax3, experimental_average_maker(detector), simulatated_average_maker(detector), sector_tuples=sector_tuples, letter = "(c)")
    profiles(ax6, experimental_average_maker(smeared_detector), simulatated_average_maker(smeared_detector), sector_tuples=sector_tuples, letter = "(f)")
    profiles(ax9, experimental_average_maker(saxs_detector), simulatated_average_maker(saxs_detector), sector_tuples=sector_tuples, letter = "(i)")

    share_lims([ax1, ax4, ax7], ax_getter= lambda ax: ax.get_xlim(), ax_setter=lambda ax, lims : ax.set_xlim(left=lims[0], right=lims[1]))
    share_lims([ax2, ax5, ax8], ax_getter= lambda ax: ax.get_xlim(), ax_setter=lambda ax, lims : ax.set_xlim(left=lims[0], right=lims[1]))
    share_lims([ax3, ax6, ax9], ax_getter= lambda ax: ax.get_xlim(), ax_setter=lambda ax, lims : ax.set_xlim(left=lims[0], right=lims[1]))
    

    

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.xaxis.set_ticklabels([])


    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        ax.set_box_aspect(1)

    ax7.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    ax8.set_xlabel(r'X (nm)',fontsize =  FONT_SIZE)
    ax9.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)

    ax3.text(0.01, 1, r'0$^{o}$', color = 'b',fontsize =  FONT_SIZE)
    ax3.text(0.01, 200, r'30$^{o}$', color = 'r',fontsize =  FONT_SIZE)
    ax3.text(0.01, 20000, r'60$^{o}$', color = 'k',fontsize =  FONT_SIZE)
    ax3.text(0.01, 1000000, r'90$^{o}$', color = 'g',fontsize =  FONT_SIZE)


    ax6.text(0.01, 2, r'0$^{o}$', color = 'b',fontsize =  FONT_SIZE)
    ax6.text(0.01, 200, r'30$^{o}$', color = 'r',fontsize =  FONT_SIZE)
    ax6.text(0.01, 20000, r'60$^{o}$', color = 'k',fontsize =  FONT_SIZE)
    ax6.text(0.01, 2000000, r'90$^{o}$', color = 'g',fontsize =  FONT_SIZE)

    ax9.text(0.01, 5000, r'0$^{o}$', color = 'b',fontsize =  FONT_SIZE)
    ax9.text(0.01, 500000, r'30$^{o}$', color = 'r',fontsize =  FONT_SIZE)
    ax9.text(0.01, 50000000, r'60$^{o}$', color = 'k',fontsize =  FONT_SIZE)
    ax9.text(0.01, 5000000000, r'90$^{o}$', color = 'g',fontsize =  FONT_SIZE)

    fig.tight_layout()

    fig.savefig(file_maker("particle_positions", ".eps"))
    fig.savefig(file_maker("particle_positions", ".pdf"))

def numpy_to_sets(numpy: np.ndarray):
    cycles = np.unique(numpy[:,0])
    get_temperature = np.frompyfunc(lambda cycle : np.average(numpy[:,4][numpy[:,0] == cycle]), 1, 1)
    get_chi_squared = np.frompyfunc(lambda cycle : (numpy[:,2][numpy[:,0] == cycle])[-1], 1, 1)
    success_rate = lambda cycle : sum(numpy[:,5][numpy[:,0] == cycle]) / len(numpy[:,5][numpy[:,0] == cycle])
    get_success_rate = np.frompyfunc(lambda cycle : len([numpy[:,5][numpy[:,0] == cycle] == 1]) / len(numpy[:,5][numpy[:,0] == cycle]), 1, 1)
    get_success_rate = np.frompyfunc(success_rate, 1, 1)
    return cycles, get_temperature(cycles), get_chi_squared(cycles), get_success_rate(cycles)



def figure_algorithm_performance():
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.set_size_inches((4 * 3,3 * 1))

    greedy = np.genfromtxt(
        r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Simulation performance greedy.dat",
        skip_header=1,
    )
    cauchy = np.genfromtxt(
        r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Simulation performance Cauchy.dat",
        skip_header=1,
    )
    very_fast = np.genfromtxt(
        r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Simulation performance Very fast.dat",
        skip_header=1,
    )

    greedy_cycles, greedy_temperature, greedy_chi_squred, greedy_success = numpy_to_sets(greedy)
    cauchy_cycles, cauchy_temperature, cauchy_chi_squared, cauchy_success = numpy_to_sets(cauchy)
    very_fast_cycles, very_fast_temperature, very_fast_chi_squared, very_fast_success = numpy_to_sets(very_fast)

    ax1.semilogy(cauchy_cycles[cauchy_temperature != 0], cauchy_temperature[cauchy_temperature != 0], 'g-')
    ax1.semilogy(very_fast_cycles[very_fast_temperature != 0], very_fast_temperature[very_fast_temperature != 0], 'r-')
    ax1.set_ylabel(r'Annealing temperature',fontsize =  FONT_SIZE)#'x-large')
    ax1.set_xlabel(r'Cycle Number',fontsize =  FONT_SIZE)#'x-large')
    down1, up1 = ax1.get_ylim()
    ax1.set_ylim(down1, up1 ** 1.5)
    ax1.text(0.05, 0.92, '(a)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(70, 0.3, r'Fast', color = 'g',fontsize = FONT_SIZE)
    ax1.text(70,0.01, r'Very fast', color = 'r',fontsize = FONT_SIZE)
    

    ax2.semilogy(greedy_cycles, greedy_chi_squred, 'b.')
    ax2.semilogy(cauchy_cycles, cauchy_chi_squared, 'g.')
    ax2.semilogy(very_fast_cycles, very_fast_chi_squared, 'r.')
    ax2.axvline(x=100, color = 'gray')
    ax2.set_ylabel(r'Reduced chi-squared',fontsize =  FONT_SIZE)#'x-large')
    ax2.set_xlabel(r'Cycle Number',fontsize =  FONT_SIZE)#'x-large')
    down2, up2 = ax2.get_ylim()
    ax2.set_ylim(down2, up2 ** 1.1)
    ax2.text(0.05, 0.92, '(b)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
    

    ax2.text(140, 5, r'Fast', color = 'g',fontsize = FONT_SIZE)
    ax2.text(35,20, r'Very fast', color = 'r',fontsize = FONT_SIZE)
    ax2.text(20, 3, r'Greedy', color = 'b',fontsize = FONT_SIZE)
    ax2.text(0.50, 0.92, r'$\longrightarrow$T = 0', color = 'gray',fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)

    ax3.plot(greedy_cycles, greedy_success, 'b.')
    ax3.plot(cauchy_cycles, cauchy_success, 'g.')
    ax3.plot(very_fast_cycles, very_fast_success, 'r.')
    ax3.axvline(x=100, color = 'gray')
    ax3.set_ylabel(r'Success rate',fontsize =  FONT_SIZE)#'x-large')
    ax3.set_xlabel(r'Cycle Number',fontsize =  FONT_SIZE)#'x-large')
    down3, up3 = ax3.get_ylim()
    ax3.set_ylim(down3, up3 * 1.15)
    ax3.text(0.05, 0.92, '(c)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes)
    

    ax3.text(50, 0.95, r'Fast', color = 'g',fontsize = FONT_SIZE)
    ax3.text(5,0.44, r'Very fast', color = 'r',fontsize = FONT_SIZE)
    ax3.text(15, 0.25, r'Greedy', color = 'b',fontsize = FONT_SIZE)
    ax3.text(0.50, 0.92, r'$\longrightarrow$T = 0', color = 'gray',fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes)


    fig.tight_layout()

    fig.savefig(file_maker("algorithm_performance", ".eps"))
    fig.savefig(file_maker("algorithm_performance", ".pdf"))

def sector_subfigure(ax, detector_up: SimulatedDetectorImage, detector_down: SimulatedDetectorImage):
    unpolarized_intensity = (detector_up.experimental_intensity + detector_down.experimental_intensity) /2
    fn_q, fn_intensity = radial_average(unpolarized_intensity, detector_down.qX, detector_down.qY, sector = (PI/2, PI/10))
    
    up_q, up_intensiyt = radial_average(detector_up.experimental_intensity, detector_up.qX, detector_up.qY, sector = (0, PI/10))
    down_q, down_intensiyt = radial_average(detector_down.experimental_intensity, detector_down.qX, detector_down.qY, sector = (0, PI/10))
    b_term = up_intensiyt - fn_intensity
    c_term = down_intensiyt - fn_intensity
    fm_intensity = (b_term + c_term) / 2
    cross_q = down_q
    q_up = up_q
    fnfm_intensity = (b_term - c_term) / 4


    exp_filter=  lambda arr : arr[10:-5:2]
    ax.loglog(exp_filter(fn_q), exp_filter(fn_intensity), 'b.')
    ax.loglog(exp_filter(cross_q), exp_filter(fnfm_intensity), 'r.')
    ax.loglog(exp_filter(q_up), exp_filter(fm_intensity), 'g.')
    #plt.show()


def angle_subfigure(ax, detector_up: SimulatedDetectorImage, detector_down:SimulatedDetectorImage):
    angles = np.linspace(-PI, +PI)
    q_values = [0.01, 0.02, 0.03, 0.04]
    intensity = detector_up.intensity - detector_down.intensity
    simulated_intensity = detector_up.simulated_intensity - detector_down.simulated_intensity
    intensity_uncertainty = np.sqrt(detector_down.intensity_err**2+ detector_up.intensity_err**2)
    def intensity_v_angle(q_c, angle, i_array):
        qy, qx = q_c * np.cos(angle), q_c * np.sin(angle) # 90 degrees from normal because angle is angle wrt field
        i_of_qxqy = intensity_at_qxqy(i_array, detector_up.qX, detector_up.qY, qx, qy)
        return i_of_qxqy
    for i, q in enumerate(q_values):
        intensity_angle_getter = np.frompyfunc(lambda angle_i: intensity_v_angle(q, angle_i, i_array=intensity), 1, 1)
        i_v_alpha = intensity_angle_getter(angles).astype(np.float64)
        uncertainty_angle_getter = np.frompyfunc(lambda angle_i: intensity_v_angle(q, angle_i, i_array=intensity_uncertainty), 1, 1)
        err_v_alpha = uncertainty_angle_getter(angles).astype(np.float64)
        #if np.max(i_v_alpha) > np.max(err_v_alpha):
        err_filt = lambda arr : arr[err_v_alpha < np.max(intensity)]
        ax.plot(angles, i_v_alpha, ['r.', 'b.', 'g.', 'k.'][i], label = [r'0.01 $\AA^{-1}$',r'0.02 $\AA^{-1}$', r'0.03 $\AA^{-1}$', r'0.04 $\AA^{-1}$' ][i])
        y_bot, y_top = ax.get_ylim()
        ax.errorbar(err_filt(angles), err_filt(i_v_alpha), yerr = err_filt(err_v_alpha), fmt = 'none', ecolor = 'lightgray', capsize = 5)
        ax.set_ylim(bottom = y_bot, top = y_top)

    for i, q in enumerate(q_values):
        intensity_angle_getter = np.frompyfunc(lambda angle_i: intensity_v_angle(q, angle_i, i_array=simulated_intensity), 1, 1)
        i_v_alpha = intensity_angle_getter(angles).astype(np.float64)
        ax.plot(angles, i_v_alpha, ['r-', 'b-', 'g-', 'k-'][i])

    ax.set_xlabel(r'Angle (rad)',fontsize =  FONT_SIZE)#'x-large')
    ax.set_ylabel(r'I$_{\uparrow}-$I$_{\downarrow}$ (cm$^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    ax.set_xticks([-PI, -PI/2, 0, +PI/2, +PI])
    ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$','0', r'$\pi/2$', r'$\pi$'])
    ax.legend(loc = "upper right", fontsize = 'x-small')
    x_min, x_max = ax.get_xlim()
    ax.set_xlim((x_min, PI + PI/2))

def figure_polarization():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    detector_up = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Programming\SasRMC\data\results\F20_Up_simulated_revised.txt",
        skip_header=1,
    )

    detector_down = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Programming\SasRMC\data\results\F20_Down_simulated_revised.txt",
        skip_header=1,
    )

    intensity_difference = detector_up.intensity - detector_down.intensity
    simulated_intensity_difference = detector_up.simulated_intensity - detector_down.simulated_intensity
    intensity_unpol = (detector_up.intensity + detector_down.intensity) / 2
    simulated_intensity_unpol = (detector_up.simulated_intensity + detector_down.simulated_intensity) / 2

    LEVELS = 30
    CMAP = 'jet'

    def detector_to_contour(ax, detector, intensity = None, show_x_label = True, show_y_label = True):
        intensity_matrix = intensity_matrix_from_detector(detector) if intensity is None else intensity
        ax.contourf(detector.qX, detector.qY, intensity_matrix, levels = LEVELS, cmap = CMAP)
        if show_x_label:
            ax.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
        else:
            ax.xaxis.set_ticklabels([])
        if show_y_label:
            ax.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
        else:
            ax.yaxis.set_ticklabels([])
        ax.text(0.025, 0.05, "Experiment",fontsize =  FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.text(0.975, 0.05, "Simulation",fontsize =  FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    

    def detector_pieces_to_intensity(detector_template, experimental_intensity, simulated_intensity):
        intensity_matrix = np.where(
            detector_template.qX < 0, 
            experimental_intensity * detector_template.shadow_factor, 
            simulated_intensity * detector_template.shadow_factor)
        return np.log(intensity_matrix)


    detector_to_contour(ax1, detector_up, show_x_label=False, show_y_label=True)
    ax1.text(0.05, 0.92, r'(a) I$\uparrow$', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    
    detector_to_contour(ax2, detector_down, show_x_label=False, show_y_label=False)
    ax2.text(0.05, 0.92, r'(b) I$\downarrow$', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
    
    detector_to_contour(ax3, detector_up, detector_pieces_to_intensity(detector_up, intensity_difference, simulated_intensity_difference ),  show_x_label=True, show_y_label=True)
    ax3.text(0.05, 0.92, r'(c) I$\uparrow-$I$\downarrow$', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes)
    
    detector_to_contour(ax4, detector_up, detector_pieces_to_intensity(detector_up, intensity_unpol, simulated_intensity_unpol ), show_x_label=True, show_y_label=False)
    ax4.text(0.05, 0.92, r'(d) (I$\uparrow+$I$\downarrow$)/2', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes)
    
    
    ax2.annotate('B', xy=(1.15,0.75), xycoords='axes fraction', xytext=(1.15, +0.25),#xytext=(0.05, 0.05),
            arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'},
            va='center', annotation_clip = False,fontsize = FONT_SIZE,ha = 'center')#, transform = ax4.transAxes)
    
    fig.set_size_inches((4 * 2,3.5 * 2))


    for ax in (ax1, ax2, ax3, ax4):
        ax.set_box_aspect(1)

    fig.tight_layout()



    
    fig.savefig(file_maker("polarization_fig", ".eps"))
    fig.savefig(file_maker("polarization_fig", ".pdf"))


def row_to_particle(row: pd.Series) -> Particle:
    if row['Particle type'] == 'CoreShellParticle':
        return CoreShellParticle.gen_from_parameters(
            position=Vector(float(row['Position.X']), float(row['Position.Y']), float(row['Position.Z'])),
            magnetization=Vector(float(row['Magnetization.X']),float(row['Magnetization.Y']),float(row['Magnetization.Z'])),
            core_radius=float(row['Core radius']),
            thickness=float(row['Shell thickness']),
            core_sld=float(row['Core SLD']),
            shell_sld=float(row['Shell SLD']),
            solvent_sld=float(row['Solvent SLD'])
        )
    return None

def dataframe_to_particle_list(data_frame: pd.DataFrame) -> List[Particle]:
    return [row_to_particle(row) for _, row in data_frame.iterrows()]



def read_simulation_output(file_path: Path) -> Tuple[List[SimulatedDetectorImage], List[Box], SimulationParams]:
    data_dict = pd.read_excel(
        file_path,
        sheet_name=None,
        keep_default_na=False
    )
    detector_list = []
    for i in range(100_000):
        key = f'Final detector image {i}'
        if key not in data_dict:
            break
        else:
            detector_list.append(SimulatedDetectorImage.gen_from_pandas(
                dataframe=data_dict[key]))
    box_list = []
    for j in range(100_000):
        key = f'Box {j} Final Particle States'
        if key not in data_dict:
            break
        else:
            dataframe = data_dict[key]
            particle_list = dataframe_to_particle_list(dataframe)
            box_list.append(box_from_detector(detector_list, particle_list))
    simulation_data = data_dict['Simulation Data']
    nuclear_rescale, magnetic_rescale = 1,1
    for _, simulation_row in simulation_data.iterrows():
        if simulation_row['Acceptable Move'] == 'TRUE' or simulation_row['Acceptable Move'] == True:
            #print(simulation_row['Acceptable Move'])
            if is_float(simulation_row['Nuclear rescale']) and float(simulation_row['Nuclear rescale']):
                nuclear_rescale = float(simulation_row['Nuclear rescale'])
            if is_float(simulation_row['Magnetic rescale']) and float(simulation_row['Magnetic rescale']):
                magnetic_rescale = float(simulation_row['Magnetic rescale'])
    simulation_params = SimulationParams(
        params=[
            SimulationParam(value = nuclear_rescale, name = "Nuclear rescale"),
            SimulationParam(value = magnetic_rescale, name = "Magnetic rescale")]
    )
    return detector_list, box_list, simulation_params

def figure_polarization_v2():

    fig, (ax1, ax2) = plt.subplots(1, 2) #plt.subplots(1,3)
    fig.set_size_inches((4 * 2,3 * 1))
    _, box_list, simulation_params = read_simulation_output(r"J:\Uni\Programming\SasRMC\data\results\20220814174915_this_is_an_example.xlsx")
    #qx_large, qy_large = np.meshgrid(np.linspace(-0.5, +0.5, 400),np.linspace(-0.5, +0.5, 400))
    config_20m_down = DetectorConfig(
        detector_distance_in_m=20,
        collimation_distance_in_m=20,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*10e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        polarization=Polarization.SPIN_DOWN
    )
    detector_20_down = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35808-M3-h-toluene-3000mT-polDown-KWS1.DAT",
        detector_config=config_20m_down,
        transpose=True
    )
    config_20m_up = DetectorConfig(
        detector_distance_in_m=20,
        collimation_distance_in_m=20,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*10e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        polarization=Polarization.SPIN_UP
    )
    detector_20_up = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35809-M3-h-toluene-3000mT-polUp-KWS1.DAT",
        detector_config=config_20m_up,
        transpose=True
    )
    config_8_down = DetectorConfig(
        detector_distance_in_m=8,
        collimation_distance_in_m=8,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*10e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        polarization=Polarization.SPIN_DOWN
    )
    detector_8_down = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35810-M3-h-toluene-3000mT-polDown-KWS1.DAT",
        detector_config=config_8_down,
        transpose=True
    )
    config_8_up = DetectorConfig(
        detector_distance_in_m=8,
        collimation_distance_in_m=8,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*10e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        polarization=Polarization.SPIN_UP
    )
    detector_8_up = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35811-M3-h-toluene-3000mT-polUp-KWS1.DAT",
        detector_config=config_8_up,
        transpose=True
    )
    config_2_down = DetectorConfig(
        detector_distance_in_m=1.5,
        collimation_distance_in_m=8,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*10e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        polarization=Polarization.SPIN_DOWN
    )
    detector_2_down = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35812-M3-h-toluene-3000mT-polDown-KWS1.DAT",
        detector_config=config_2_down,
        transpose=True
    )
    config_2_up = DetectorConfig(
        detector_distance_in_m=1.5,
        collimation_distance_in_m=8,
        collimation_aperture_area_in_m2=30e-3*30e-3,
        sample_aperture_area_in_m2=6e-3*10e-3,
        detector_pixel_size_in_m=5.6e-3,
        wavelength_in_angstrom=5,
        polarization=Polarization.SPIN_UP
    )
    detector_2_up = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35813-M3-h-toluene-3000mT-polUp-KWS1.DAT",
        detector_config=config_2_up,
        transpose=True
    )
    toluene_20 = DetectorImage.gen_from_txt(r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35702-h-toluene-pol-KWS1.DAT", transpose=True)
    toluene_8 = DetectorImage.gen_from_txt(r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35704-h-toluene-pol-KWS1.DAT", transpose=True)
    toluene_2 = DetectorImage.gen_from_txt(r"J:\Uni\Sorted results\User Experiments\p14961 - Nandakumaran\Reduced\Polarized\ASCII-I\I-SM-35706-h-toluene-pol-KWS1.DAT", transpose=True)
    subtract_buffer_intensity(detector_20_down, toluene_20)
    subtract_buffer_intensity(detector_20_up, toluene_20)
    subtract_buffer_intensity(detector_8_down, toluene_8)
    subtract_buffer_intensity(detector_8_up, toluene_8)
    subtract_buffer_intensity(detector_2_down, toluene_2)
    subtract_buffer_intensity(detector_2_up, toluene_2)
    rescale_factor, magnetic_rescale = [simulation_param.value for simulation_param in simulation_params.params]
    for detector in [detector_20_down, detector_20_up, detector_8_down, detector_8_up, detector_2_down, detector_2_up]:
        print(detector.qY.shape)
        analytical_calculator = AnalyticalCalculator(detector.qX, detector.qY)
        simulated_intensity = box_intensity_average(box_list, analytical_calculator, rescale_factor, magnetic_rescale, detector.polarization)
        smeared_intensity = smear_simulated_intensity(simulated_intensity, analytical_calculator.qx_array, analytical_calculator.qy_array, detector)
    
    if True:
        angle_subfigure(ax1, detector_8_up, detector_8_down)
        ax1.text(0.05, 0.92, r'(a)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    else:

        sector_subfigure(ax1, detector_20_up, detector_20_down)
        sector_subfigure(ax1, detector_8_up, detector_8_down)
        sector_subfigure(ax1, detector_2_up, detector_2_down)
        ax1.set_ylim(7.045615400141189e-07, 153.92882006237988)
    #for detector_up, detector_down in zip([detector_20_up, detector_8_up,detector_2_up ], [detector_20_down, detector_8_down,detector_2_down]):
    q, fn, cross, fm, fn_err, cross_err, fm_err = unpol_analysis(
        detector_20_up,
        detector_20_down,
        intensity_getter= lambda d: d.intensity
    )
    q_coll = np.append(q[10:-20], [])
    fn_coll = np.append(fn[10:-20], [])
    fn_err_coll = np.append(fn_err[10:-20], [])
    fm_coll = np.append(fm[10:-20], [])
    fm_err_coll = np.append(fm_err[10:-20], [])
    exp_filter = lambda arr : arr[10:-20:4]
    ax2.errorbar(
        exp_filter(q), exp_filter(fn), yerr = exp_filter(fn_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(fn), 'b.', label = 'F$_{N}^2$')
    ax2.errorbar(
        exp_filter(q), exp_filter(cross), yerr = exp_filter(cross_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(cross), 'r.', label = 'F$_{N}$F$_{M}$')
    ax2.errorbar(
        exp_filter(q), exp_filter(fm), yerr = exp_filter(fm_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(fm), 'g.', label = 'F$_{M}^2$')

    q_sim, fn_sim, cross_sim, fm_sim, fn_sim_err, cross_sim_err, fm_sim_err = unpol_analysis(
        detector_20_up,
        detector_20_down,
        intensity_getter= lambda d: d.simulated_intensity
    )
    sim_filter = lambda arr: arr[10:-20]
    ax2.loglog(sim_filter(q_sim), sim_filter(fn_sim), 'b-')
    ax2.loglog(sim_filter(q_sim), sim_filter(cross_sim), 'r-')
    ax2.loglog(sim_filter(q_sim), sim_filter(fm_sim), 'g-')
    q, fn, cross, fm, fn_err, cross_err, fm_err = unpol_analysis(
        detector_8_up,
        detector_8_down,
        intensity_getter= lambda d: d.intensity
    )
    q_coll = np.append(q_coll, q[10:-10])
    fn_coll = np.append(fn_coll, fn[10:-10])
    fn_err_coll = np.append(fn_err_coll, fn_err[10:-10])
    fm_coll = np.append(fm_coll, fm[10:-10])
    fm_err_coll = np.append(fm_err_coll, fm_err[10:-10])
    exp_filter = lambda arr : arr[10:-6:4]
    ax2.errorbar(
        exp_filter(q), exp_filter(fn), yerr = exp_filter(fn_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(fn), 'b.')#, label = 'F$_{N}^2$')
    ax2.errorbar(
        exp_filter(q), exp_filter(cross), yerr = exp_filter(cross_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(cross), 'r.')#, label = 'F$_{N}$F$_{M}$')
    ax2.errorbar(
        exp_filter(q), exp_filter(fm), yerr = exp_filter(fm_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(fm), 'g.')#, label = 'F$_{M}^2$')

    q_sim, fn_sim, cross_sim, fm_sim, fn_sim_err, cross_sim_err, fm_sim_err = unpol_analysis(
        detector_8_up,
        detector_8_down,
        intensity_getter= lambda d: d.simulated_intensity
    )
    sim_filter = lambda arr: arr[10:-6]
    ax2.loglog(sim_filter(q_sim), sim_filter(fn_sim), 'b-')
    ax2.loglog(sim_filter(q_sim), sim_filter(cross_sim), 'r-')
    ax2.loglog(sim_filter(q_sim), sim_filter(fm_sim), 'g-')
    q, fn, cross, fm, fn_err, cross_err, fm_err  = unpol_analysis(
        detector_2_up,
        detector_2_down,
        intensity_getter= lambda d: d.intensity
    )
    q_coll = np.append(q_coll, q[10:-10])
    fn_coll = np.append(fn_coll, fn[10:-10])
    fn_err_coll = np.append(fn_err_coll, fn_err[10:-10])
    fm_coll = np.append(fm_coll, fm[10:-10])
    fm_err_coll = np.append(fm_err_coll, fm_err[10:-10])
    np.savetxt(file_maker("nuclear_form", ".txt"), np.array([(q_i, fn_i, fn_i_err) for q_i, fn_i, fn_i_err in zip(q_coll, fn_coll, fn_err_coll)]))
    np.savetxt(file_maker("magnetic_form", ".txt"), np.array([(q_i, fm_i, fm_i_err) for q_i, fm_i, fm_i_err in zip(q_coll, fm_coll, fm_err_coll)]))
    
    exp_filter = lambda arr : arr[10:-15:4]
    ax2.errorbar(
        exp_filter(q), exp_filter(fn), yerr = exp_filter(fn_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(fn), 'b.')#, label = 'F$_{N}^2$')
    ax2.errorbar(
        exp_filter(q), exp_filter(cross), yerr = exp_filter(cross_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(cross), 'r.')#, label = 'F$_{N}$F$_{M}$')
    ax2.errorbar(
        exp_filter(q), exp_filter(fm), yerr = exp_filter(fm_err), fmt = 'none', ecolor = 'lightgray', capsize = 5
    )
    ax2.loglog(exp_filter(q), exp_filter(fm), 'g.')#, label = 'F$_{M}^2$')

    q_sim, fn_sim, cross_sim, fm_sim, fn_sim_err, cross_sim_err, fm_sim_err = unpol_analysis(
        detector_2_up,
        detector_2_down,
        intensity_getter= lambda d: d.simulated_intensity
    )
    sim_filter = lambda arr: arr[10:-15]
    ax2.loglog(sim_filter(q_sim), sim_filter(fn_sim), 'b-')
    ax2.loglog(sim_filter(q_sim), sim_filter(cross_sim), 'r-')
    ax2.loglog(sim_filter(q_sim), sim_filter(fm_sim), 'g-')

    ax2.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    ax2.set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    bottom, top = ax2.get_ylim()
    ax2.set_ylim(bottom, 9 * top)
    

    ax2.text(0.05, 0.92, r'(b)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
    #ax2.text(0.05, 0.92, r'(b)', color = "red", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
    ax2.legend(loc = "lower left", fontsize = 'small')

    get_from_box_list = lambda getter_fn: np.array([getter_fn(particle) for particle in box_list[0].particles])
    pos_x = get_from_box_list(lambda particle: particle.position.x)
    pos_y = get_from_box_list(lambda particle: particle.position.y)
    mag_x = get_from_box_list(lambda particle: particle.magnetization.x)
    mag_y = get_from_box_list(lambda particle: particle.magnetization.y)
    radius = get_from_box_list(lambda particle: particle.shapes[1].radius)

    angles = get_from_box_list(lambda particle : np.abs(np.arctan2(particle.magnetization.x, particle.magnetization.y)))

    #count = np.sum(get_from_box_list(lambda particle : 1))
    
    fig2, (ax3, ax4) = plt.subplots(1, 2) #plt.subplots(1,3)
    fig2.set_size_inches((4 * 2,4 * 1))
    #pos_x, pos_y, mag_x, mag_y, radius = [particle.position.x, particle.po for particle in box_list[0].particles]magnetic_particle_configs[:,2], magnetic_particle_configs[:,3], magnetic_particle_configs[:,8], magnetic_particle_configs[:,9], magnetic_particle_configs[:,11]
    a_x, a_y = radius* mag_x / np.sqrt(mag_x**2 + mag_y**2), radius* mag_y / np.sqrt(mag_x**2 + mag_y**2)
    ax3.scatter(pos_x / 10, pos_y / 10, s = 2 * (radius/10)**1, facecolors='none', edgecolors='b')
    ARROW_MAGNIFIER = 3
    for x, y, dx, dy in zip(pos_x, pos_y, a_x, a_y):
        ax3.arrow(x/10, y/10,ARROW_MAGNIFIER * dx/10,ARROW_MAGNIFIER* dy/10,
        color= 'r', width = 3,head_width = 5 * 3)

    ax3.set_xlabel(r'X (nm)',fontsize =  FONT_SIZE)#'x-large')
    ax3.set_ylabel(r'Y (nm)',fontsize =  FONT_SIZE)#'x-large')
    
    left, right = ax3.get_xlim()
    bottom, top = ax3.get_xlim()

    new_lims = -550, +550#np.min([left, down]), np.max([up, right])

    ax3.set_xlim(new_lims[0], new_lims[1])
    ax3.set_ylim(new_lims[0], new_lims[1])
    ax3.text(0.05, 0.92, r'(a)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes)
    
    ax3.set_box_aspect(1)

    

    
    ax4.hist(angles, bins = 11, edgecolor = 'black', linewidth=1.2)
    ax4.set_ylim(0, 60)
    ax4.set_ylabel(r'Particles', fontsize = FONT_SIZE)
    ax4.set_xlabel(r'Magnetization angle (rad)', fontsize = FONT_SIZE)
    x_ticks = [i * PI/10 for i in range(6)]# + [PI/2]
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([str(x_tick / PI) + r'$\pi$' for x_tick in x_ticks])
    ax4.text(0.05, 0.92, r'(b)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes)
    ax4.set_box_aspect(1)


    fig.tight_layout()

    fig.savefig(file_maker("magnetization_fig", ".eps"))
    fig.savefig(file_maker("magnetization_fig", ".pdf"))

    
    fig2.tight_layout()

    fig2.savefig(file_maker("magnetization_dist_fig", ".eps"))
    fig2.savefig(file_maker("magnetization_dist_fig", ".pdf"))
    #plt.hist(angles, bins = 10)
    #plt.show()

def quick_test(): # Add to version control then delete
    box_writer = BoxWriter.standard_box_writer()
    box = Box(
        particles=[CoreShellParticle.gen_from_parameters(position = Vector.null_vector(), core_radius=100) for _ in range(20)],
        cube=Cube(dimension_0=10000, dimension_1=10000, dimension_2=10000)
    )
    box.force_inside_box(in_plane=True)
    fig = box_writer.to_plot(box)
    plt.show()


def main():
    #figure_form_factors()
    figure_particle_maps()
    #figure_algorithm_performance()
    #figure_polarization()
    #figure_polarization_v2()
    #quick_test()

if __name__ == "__main__":
    main()

#%%
