#%%
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize
import pandas as pd

import sas_rmc

from sas_rmc import DetectorImage, Vector, VectorSpace, SimulatedDetectorImage, DetectorConfig, Polarization
from sas_rmc.box_simulation import Box
from sas_rmc.command_writer import BoxWriter
from sas_rmc.form_calculator import box_intensity_average
from sas_rmc.particle import CoreShellParticle, NumericalParticle, NumericalParticleCustom, Particle
from sas_rmc.scattering_simulation import SimulationParam, SimulationParams
from sas_rmc.shapes import Cube
from sas_rmc.simulator_factory import is_float, subtract_buffer_intensity
from sas_rmc.scattering_simulation import detector_smearer

file_maker = sas_rmc.simulator_factory.generate_file_path_maker(Path.cwd(), description="figures")

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


def show_particle(particle: NumericalParticle):
    xy_axis = 2
    sld = particle.sld_from_vector_space()
    vector_space = particle.vector_space
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
    '''i = np.argmin(np.abs(qx_array[0:,] - qx_i))
    j = np.argmin(np.abs(qy_array[:,0] - qy_i))
    i_s = adjust_i_array([i-1, i, i+1], i, len(qx_array[0:,]))
    j_s = adjust_i_array([j -1, j, j+1], j, len(qy_array[:,0]))'''

    i_s = subset_array(qx_array[0:,], qx_i, 5)
    j_s = subset_array(qy_array[:,0], qy_i, 5)
    
    qx = [qx_array[0,ii] for ii in i_s]
    qy = [qy_array[jj,0] for jj in j_s]
    i_sub = [[intensity_array[jj, ii] for jj in j_s] for ii in i_s]
    interpolator = interpolate.interp2d(qx, qy, i_sub, copy = False)
    return np.average(interpolator(qx_i, qy_i))
    

def radial_average(i_array: np.ndarray, qx_array: np.ndarray, qy_array: np.ndarray, sector: Tuple[float, float] = (0, np.inf)):
    q_mod = np.sqrt(qx_array**2 + qy_array**2)

    '''step = np.average(np.diff(qx_array[0,:]))
    print(step)'''
    
    q = np.arange(
        start = np.min(np.where(i_array != 0, q_mod, np.inf)),
        stop = np.max(q_mod),
        step = np.average(np.diff(qx_array[0,:]))
        )
    
    #interpolator = interpolate.interp2d(qx_array, qy_array, i_array, kind='cubic')
    def i_of_q(q_c):
        angles = np.linspace(-PI, +PI, num = 180)
        sector_filter = np.abs(angles - sector[0]) < sector[1] / 2
        qx, qy = q_c * np.cos(angles)[sector_filter], q_c * np.sin(angles)[sector_filter]
        #print(q_c, np.average(np.sqrt(qx**2 + qy**2)))
        #print(len(qx))
        if len(qx) == 0:
            return 0
        
        intensities = np.frompyfunc(lambda x, y: intensity_at_qxqy(i_array, qx_array, qy_array, x, y),2,1)(qx, qy).astype(np.float64)
        #print(np.std(intensities) / np.average(intensities))
        if len(intensities[intensities != 0]) == 0:
            return 0
        return np.average(intensities[intensities > 0])
    averager = np.frompyfunc(i_of_q, 1, 1)
    return q, averager(q).astype(np.float64)

def sin_2_theta(theta, a, offset = 0):
    return a * (np.sin(theta)**2) + offset
    

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
        qx, qy = q_c * np.cos(angles), q_c * np.sin(angles)
        if len(qx) == 0:
            return 0
        fnfm_intensities = np.frompyfunc(lambda x, y: intensity_at_qxqy(i_diff_array, qx_array, qy_array, x, y),2,1)(qx, qy).astype(np.float64)
        if len(fnfm_intensities[fnfm_intensities!=0]) < 6:
            return 0, 0, 0
        popt, pcov = optimize.curve_fit(
            sin_2_theta, angles[fnfm_intensities!=0], fnfm_intensities[fnfm_intensities!=0], p0 = [np.max(fnfm_intensities)]
        )
        fnfm = popt[0] / 4

        unpol_intensities = np.frompyfunc(lambda x, y: intensity_at_qxqy(i_unpol_array, qx_array, qy_array, x, y),2,1)(qx, qy).astype(np.float64)

        popt_u, pcov_u = optimize.curve_fit(
            sin_2_theta, angles[unpol_intensities!=0], unpol_intensities[unpol_intensities!=0], p0 = [np.max(unpol_intensities), np.min(unpol_intensities)]
        )
        fn2 = popt_u[1]

        fm2 = fnfm**2 / fn2

        return fn2, fnfm, fm2


    sin_2_fitter = np.frompyfunc(fn_fm, 1, 3)
    f_nuc, cross_term, f_mag = sin_2_fitter(q)
    return q[f_nuc != 0], f_nuc.astype(np.float64)[f_nuc != 0], cross_term.astype(np.float64)[f_nuc != 0], f_mag.astype(np.float64)[f_nuc != 0]
    

FONT_SIZE = 14

def guinier_model(q, i_0, r_g):
    i =  i_0 * np.exp(-(q * r_g)**2 / 3)
    return i


def figure_form_factors():
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches((4 * 3,3.5 * 2))

    

    def onion_sld(position: Vector) -> float:
        if position.mag < 100:
            return 10
        if position.mag < 120:
            return 5
        return 0

    core_shell_particle = CoreShellParticle.gen_from_parameters(
        position=Vector.null_vector(),
        magnetization=Vector.null_vector(),
        core_radius=100,
        thickness=20,
        core_sld=10,
        shell_sld=5,
        solvent_sld=0,
    )

    def dumbbell_sld(position: Vector) -> float:
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
        return 0

    onion_particle = NumericalParticleCustom(
        get_sld_function=onion_sld,
        vector_space=VectorSpace.gen_from_bounds(
            x_min = -160, x_max= 160, x_num = 40,
            y_min = -160, y_max= 160, y_num = 40,
            z_min = -160, z_max= 160, z_num = 40,
        )
    )
    x_arr, y_arr, flat_sld = show_particle(onion_particle)

    
    
    
    axs[0,0].pcolormesh(x_arr / 10, y_arr / 10, flat_sld, shading = 'auto')
    #axs[0,0].set_xlabel('X (nm)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[0,0].set_ylabel('Y (nm)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[0,0].xaxis.set_ticklabels([])
    #axs[0,0].text(-14, +13, '(a) Core-shell particle', color = "white")
    axs[0,0].text(0.05, 0.95, '(a)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,0].transAxes)
    axs[0,0].text(0.05, 0.05, 'Core-shell particle', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,0].transAxes)
    
    detector = default_detector_image()
    qx, qy = detector.qX, detector.qY

    form_array = onion_particle.form_array(qx, qy, orientation=Vector(0,0,1))

    axs[0,1].contourf(qx, qy, mod(form_array), levels = 20)
    #axs[0,1].set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[0,1].set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[0,1].xaxis.set_ticklabels([])
    axs[0,1].text(0.05, 0.95, '(b)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,1].transAxes)
    
    qx_large, qy_large = np.meshgrid(np.linspace(-0.5, +0.5, 400),np.linspace(-0.5, +0.5, 400))
    form_array_2 = onion_particle.form_array(qx_large, qy_large, orientation=Vector(0,0,1))
    i_array = mod(form_array_2)
    #q = np.linspace(2e-3, 4e-1, num  = 300)
    
    
    q, i_1d = radial_average(i_array, qx_large, qy_large)#, q)
    q = q[i_1d != 0]
    i_1d = i_1d[i_1d != 0]
    i_1d = i_1d[q < 4e-1]
    q = q[q < 4e-1]

    i_core_shell = mod(core_shell_particle.form_array(q, np.zeros(q.shape), orientation=Vector.null_vector()))


    axs[0,2].loglog(q, i_core_shell / np.max(i_core_shell), 'r-', label = "Analytical")
    axs[0,2].loglog(q, i_1d / np.max(i_1d), 'b.', label = "Numerical")
    #axs[0,2].set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    axs[0,2].set_ylabel(r'I/I$_{0}$',fontsize =  FONT_SIZE)
    axs[0,2].xaxis.set_ticklabels([])
    bottom, top = axs[0,2].get_ylim()
    axs[0,2].set_ylim(bottom, top * 4)
    axs[0,2].text(0.05, 0.95, '(c)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[0,2].transAxes)
    axs[0,2].legend(loc = 'upper right')

    #axs[0,0].imshow()

    dumbell_particle = NumericalParticleCustom(
        get_sld_function=dumbbell_sld,
        vector_space=VectorSpace.gen_from_bounds(
            x_min = -160, x_max= 160, x_num = 40,
            y_min = -160, y_max= 160, y_num = 40,
            z_min = -160, z_max= 160, z_num = 40,
        )
    )
    x_arr, y_arr, flat_sld = show_particle(dumbell_particle)
    axs[1,0].pcolormesh(x_arr / 10, y_arr / 10, flat_sld, shading = 'auto')
    axs[1,0].set_xlabel('X (nm)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[1,0].set_ylabel('Y (nm)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[1,0].text(0.05, 0.95, '(d)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,0].transAxes)
    axs[1,0].text(0.05, 0.05, 'Dumbbell particle', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,0].transAxes)
    
    form_array = dumbell_particle.form_array(qx, qy, orientation=Vector(0,0,1))

    axs[1,1].contourf(qx, qy, mod(form_array), levels = 20)
    axs[1,1].set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize=10)
    axs[1,1].set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#, fontsize='medium')
    axs[1,1].text(0.05, 0.95, '(e)', color = "white", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=axs[1,1].transAxes)
    
    #qx_large, qy_large = np.meshgrid(np.linspace(-0.5, +0.5, 200),np.linspace(-0.5, +0.5, 200))
    form_array_2 = dumbell_particle.form_array(qx_large, qy_large, orientation=Vector(0,0,1))
    i_array = mod(form_array_2)

    

    q = np.linspace(2e-3, 4e-1, num  = 300)
    
    
    q, i_1d = radial_average(i_array, qx_large, qy_large)#, q)
    q = q[i_1d != 0]
    i_1d = i_1d[i_1d != 0]
    i_1d = i_1d[q < 4e-1]
    q = q[q < 4e-1]

    popt, pcov = optimize.curve_fit(guinier_model, q, i_1d /np.max(i_1d), p0= [1, 100], method='lm')
    print(popt, pcov)

    fitted_i_0, fitted_rg = popt

    axs[1,2].loglog(q[q * fitted_rg < (1.3 * 3)], guinier_model(q, fitted_i_0, fitted_rg)[q * fitted_rg < (1.3 * 3)], 'r-', label = "Guinier")
    axs[1,2].loglog(q, i_1d / np.max(i_1d), 'b.', label = "Numerical")
    axs[1,2].set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    axs[1,2].set_ylabel(r'I/I$_{0}$',fontsize =  FONT_SIZE)
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

def figure_particle_maps():
    #from matplotlib.gridspec import GridSpec

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    
    fig.set_size_inches((4 * 3,3.5 * 2))

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
    

    
    for c, angle, factor in zip(['b', 'r', 'k', 'g'], [0, PI/6, 2*PI/6, 3*PI/6], [10**0, 10**2, 10**4, 10**6]):
        q_unsmeared, radial_average_unsmeared = radial_average(
            detector.experimental_intensity * detector.shadow_factor,
            detector.qX,
            detector.qY,
            sector = (angle, PI / 10)
        )
        '''q_unsmeared = q_unsmeared[radial_average_unsmeared > 0]
        radial_average_smeared = radial_average_unsmeared[radial_average_unsmeared > 0]'''
        ax3.loglog(
            q_unsmeared[radial_average_unsmeared > 0], factor * radial_average_unsmeared[radial_average_unsmeared > 0], c + '.'
        )
        q_unsmeared, radial_average_unsmeared = radial_average(
            detector.simulated_intensity * detector.shadow_factor,
            detector.qX,
            detector.qY,
            sector = (angle, PI / 10)
        )
        '''q_unsmeared = q_unsmeared[radial_average_unsmeared > 0]
        radial_average_smeared = radial_average_unsmeared[radial_average_unsmeared > 0]'''
        ax3.loglog(
            q_unsmeared[radial_average_unsmeared > 0], factor * radial_average_unsmeared[radial_average_unsmeared > 0], c + '-'
        )
    #ax3.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  16)
    ax3.set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)
    ax3.text(0.05, 0.95, '(c)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes)
    
    #radial_average(detector.experimental_intensity, detector.qX, detector.qY, 

    #ax1 = fig.add_subplot(gs[0,0])
    ax1.contourf(detector.qX, detector.qY, intensity_matrix, levels = levels, cmap = 'jet')
    #ax1.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
    ax1.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    ax1.text(0.025, 0.05, "Experiment",fontsize =  FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.975, 0.05, "Simulation",fontsize =  FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes)
    #ax1.text(+0.005, -0.025, "Simulation",fontsize =  FONT_SIZE)
    ax1.text(0.975, 0.95, "No smearing", color = "black", fontsize = FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.05, 0.95, '(a)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    
    #ax2 = fig.add_subplot(gs[0,1])

    '''axs[0,1].plot(unsmeared_positions[:,0] /10, unsmeared_positions[:,1] /10, 'b.')
    axs[0,1].set_xlabel(r'X (nm)',fontsize =  16)
    axs[0,1].set_ylabel(r'Y (nm)',fontsize =  16)'''
    ax2.plot(unsmeared_positions[:,0] /10, unsmeared_positions[:,1] /10, 'b.')
    #ax2.set_xlabel(r'X (nm)',fontsize =  16)
    ax2.set_ylabel(r'Y (nm)',fontsize =  FONT_SIZE)
    ax2.text(0.05, 0.95, '(b)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
    
    
    '''axs[0,0].contourf(detector.qX, detector.qY, np.log(intensity_matrix), levels = 30, cmap = 'jet')
    axs[0,0].set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
    axs[0,0].set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  16)#'x-large')
    axs[0,0].text(-0.025, -0.025, "Experiment",fontsize =  14)
    axs[0,0].text(+0.005, -0.025, "Simulation",fontsize =  14)
    axs[0,0].text(-0.025, +0.023, "(a)",fontsize =  14)'''

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_ticklabels([])

    smeared_positions = np.genfromtxt(
        fname = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Smeared_particle_positions_200.txt",
        skip_header = 1
    )
    detector = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Smeared_detector_200.txt",
        skip_header=1,
    )
    
    #intensity_matrix = np.where(detector.qX < 0, detector.experimental_intensity, detector.simulated_intensity)
    
    

    #ax3 = fig.add_subplot(gs[1, 0])

    ax4.contourf(smeared_detector.qX, smeared_detector.qY, smeared_intensity_matrix, levels = levels, cmap = 'jet')
    ax4.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    ax4.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    ax4.text(0.025, 0.05, "Experiment",fontsize =  FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes)
    ax4.text(0.975, 0.05, "Simulation",fontsize =  FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax4.transAxes)
    #ax4.text(-0.025, +0.023, "With smearing",fontsize =  FONT_SIZE)
    ax4.text(0.975, 0.95, "With smearing", color = "black", fontsize = FONT_SIZE,horizontalalignment='right', verticalalignment='center', transform=ax4.transAxes)
    ax4.text(0.05, 0.95, '(d)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes)

    #ax4 = fig.add_subplot(gs[1, 1])
    ax5.plot(smeared_positions[:,0] /10, smeared_positions[:,1] /10, 'b.')
    ax5.set_xlabel(r'X (nm)',fontsize =  FONT_SIZE)
    ax5.set_ylabel(r'Y (nm)',fontsize =  FONT_SIZE)
    ax5.text(0.05, 0.95, '(e)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax5.transAxes)
    
    for ax in ax2, ax5:
        ax.set_xlim(-14070.66526562795/20, +14070.66526562795/20)
        ax.set_ylim(-14070.66526562795/20, +14070.66526562795/20)
  
    for c, angle, factor in zip(['b', 'r', 'k', 'g'], [0, PI/6, 2*PI/6, 3*PI/6], [10**0, 10**2, 10**4, 10**6]):
        q_smeared, radial_average_smeared = radial_average(
            smeared_detector.experimental_intensity * smeared_detector.shadow_factor,
            smeared_detector.qX,
            smeared_detector.qY,
            sector = (angle, PI / 10)
        )
        q_smeared = q_smeared[radial_average_smeared > 0]
        radial_average_smeared = radial_average_smeared[radial_average_smeared > 0]
        ax6.loglog(
            q_smeared, factor * radial_average_smeared, c + '.'
        )
        
        q_smeared, radial_average_smeared = radial_average(
            smeared_detector.simulated_intensity * smeared_detector.shadow_factor,
            smeared_detector.qX,
            smeared_detector.qY,
            sector = (angle, PI / 10)
        )
        q_smeared = q_smeared[radial_average_smeared > 0]
        radial_average_smeared = radial_average_smeared[radial_average_smeared > 0]
        ax6.loglog(
            q_smeared, factor * radial_average_smeared, c + '-'
        )
        
    ax6.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)
    ax6.set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)
    ax6.text(0.05, 0.95, '(f)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax6.transAxes)
    
    left3, right3 = ax3.get_xlim()
    left6, right6 = ax6.get_xlim()
    down3, up3 = ax3.get_ylim()
    down6, up6 = ax6.get_ylim()
    for i, ax in enumerate([ax3, ax6]):
        ax.set_xlim(min(left3, left6), max(right3, right6))
        ax.set_ylim(min(down3, down6), max(up3, up6))
        ax.text(0.025, 0.05, ["No smearing", "With smearing"][i], color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_box_aspect(1)

    ax3.text(0.01, 1, r'0$^{o}$', color = 'b',fontsize =  FONT_SIZE)
    ax3.text(0.01, 200, r'30$^{o}$', color = 'r',fontsize =  FONT_SIZE)
    ax3.text(0.01, 20000, r'60$^{o}$', color = 'k',fontsize =  FONT_SIZE)
    ax3.text(0.01, 1000000, r'90$^{o}$', color = 'g',fontsize =  FONT_SIZE)


    ax6.text(0.01, 2, r'0$^{o}$', color = 'b',fontsize =  FONT_SIZE)
    ax6.text(0.01, 200, r'30$^{o}$', color = 'r',fontsize =  FONT_SIZE)
    ax6.text(0.01, 20000, r'60$^{o}$', color = 'k',fontsize =  FONT_SIZE)
    ax6.text(0.01, 2000000, r'90$^{o}$', color = 'g',fontsize =  FONT_SIZE)

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


def angle_subfigure(ax, detector_up: SimulatedDetectorImage, detector_down:SimulatedDetectorImage):
    angles = np.linspace(-PI, +PI)
    q_values = [0.01, 0.02, 0.03, 0.04]
    intensity = detector_up.intensity - detector_down.intensity
    simulated_intensity = detector_up.simulated_intensity - detector_down.simulated_intensity
    def intensity_v_angle(q_c, angle, i_array):
        qy, qx = q_c * np.cos(angle), q_c * np.sin(angle) # 90 degrees from normal because angle is angle wrt field
        i_of_qxqy = intensity_at_qxqy(i_array, detector_up.qX, detector_up.qY, qx, qy)
        return i_of_qxqy
    for i, q in enumerate(q_values):
        intensity_angle_getter = np.frompyfunc(lambda angle_i: intensity_v_angle(q, angle_i, i_array=intensity), 1, 1)
        i_v_alpha = intensity_angle_getter(angles).astype(np.float64)
        ax.plot(angles, i_v_alpha, ['r.', 'b.', 'g.', 'k.'][i], label = [r'0.01 $\AA^{-1}$',r'0.02 $\AA^{-1}$', r'0.03 $\AA^{-1}$', r'0.04 $\AA^{-1}$' ][i])

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
        file_location=r'J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\F20_Up_simulated.txt',
        skip_header=1,
    )

    detector_down = SimulatedDetectorImage.gen_from_txt(
        file_location=r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\F20_Down_simulated.txt",
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


    fig_2, (ax5,ax6,ax7) = plt.subplots(1,3)
    fig_2.set_size_inches((3.5 * 3,3 * 1))

    q, fn, cross, fm = unpol_analysis(
        detector_up,
        detector_down,
        intensity_getter= lambda d: d.intensity
    )

    q_sim, fn_sim, cross_sim, fm_sim = unpol_analysis(
        detector_up,
        detector_down,
        intensity_getter= lambda d: d.simulated_intensity
    )

    
    
    angle_subfigure(ax5, detector_up, detector_down)
    ax5.text(0.05, 0.92, r'(a)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax5.transAxes)
    

    ax6.loglog(
        q, fn, 'b.'
    )
    ax6.loglog(
        q, cross, 'r.'
    )
    ax6.loglog(
        q, fm, 'g.'
    )
    ax6.loglog(
        q_sim, fn_sim, 'b-'
    )
    ax6.loglog(
        q_sim, cross_sim, 'r-'
    )
    ax6.loglog(
        q_sim, fm_sim, 'g-'
    )

    ax6.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    ax6.set_ylabel(r'Intensity (cm$^{-1}$)',fontsize =  FONT_SIZE)#'x-large')
    bottom, top = ax6.get_ylim()
    ax6.set_ylim(bottom, 9 * top)

    ax6.text(0.05, 0.92, r'(b)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax6.transAxes)
    
    #fig_2.set_size_inches((4 * 3,3 * 1))

    magnetic_particle_configs = np.genfromtxt(
        fname = r"J:\Uni\Programming\SANS_Numerical_Simulation_v2\Results\Analyzed data\Final_magnetic_particle_states.txt",
        skip_header = 1
    )
    pos_x, pos_y, mag_x, mag_y, volume = magnetic_particle_configs[:,2], magnetic_particle_configs[:,3], magnetic_particle_configs[:,8], magnetic_particle_configs[:,9], magnetic_particle_configs[:,11]
    radius = (3 * volume / (4*PI)) ** (1/3)
    a_x, a_y = radius* mag_x / np.sqrt(mag_x**2 + mag_y**2), radius* mag_y / np.sqrt(mag_x**2 + mag_y**2)
    ax7.scatter(pos_x / 10, pos_y / 10, s = 2 * (radius/10)**1, facecolors='none', edgecolors='b')
    ARROW_MAGNIFIER = 3
    for x, y, dx, dy in zip(pos_x, pos_y, a_x, a_y):
        ax7.arrow(x/10, y/10,ARROW_MAGNIFIER * dx/10,ARROW_MAGNIFIER* dy/10,
        color= 'r', width = 3,head_width = 5 * 3)

    ax7.set_xlabel(r'X (nm)',fontsize =  FONT_SIZE)#'x-large')
    ax7.set_ylabel(r'Y (nm)',fontsize =  FONT_SIZE)#'x-large')
    
    left, right = ax7.get_xlim()
    bottom, top = ax7.get_xlim()

    new_lims = -500, +500#np.min([left, down]), np.max([up, right])

    ax7.set_xlim(new_lims[0], new_lims[1])
    ax7.set_ylim(new_lims[0], new_lims[1])
    ax7.text(0.05, 0.92, r'(c)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax7.transAxes)
    
    ax7.set_box_aspect(1)

    fig_2.tight_layout()

    
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

def read_simulation_output(file_path: Path) -> Tuple[List[DetectorImage], List[Box], SimulationParams]:
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
            detector_list.append(DetectorImage.gen_from_pandas(
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

    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.set_size_inches((3.5 * 3,3 * 1))
    _, box_list, simulation_params = read_simulation_output(r"J:\Uni\Programming\SasRMC\data\results\20220503204445_this_is_a_test.xlsx")
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
        simulated_intensity = box_intensity_average(box_list, detector.qX, detector.qY, rescale_factor, magnetic_rescale, detector.polarization)
        simulated_detector = detector_smearer(simulated_intensity, detector.qX, detector.qY, detector)
        
    angle_subfigure(ax1, detector_8_up, detector_8_down)
    ax1.text(0.05, 0.92, r'(a)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    

    #for detector_up, detector_down in zip([detector_20_up, detector_8_up,detector_2_up ], [detector_20_down, detector_8_down,detector_2_down]):
    q, fn, cross, fm = unpol_analysis(
        detector_20_up,
        detector_20_down,
        intensity_getter= lambda d: d.intensity
    )
    exp_filter = lambda arr : arr[10:-20:4]
    ax2.loglog(exp_filter(q), exp_filter(fn), 'b.', label = 'F$_{N}^2$')
    ax2.loglog(exp_filter(q), exp_filter(cross), 'r.', label = 'F$_{N}$F$_{M}$')
    ax2.loglog(exp_filter(q), exp_filter(fm), 'g.', label = 'F$_{M}^2$')

    q_sim, fn_sim, cross_sim, fm_sim = unpol_analysis(
        detector_20_up,
        detector_20_down,
        intensity_getter= lambda d: d.simulated_intensity
    )
    sim_filter = lambda arr: arr[10:-20]
    ax2.loglog(sim_filter(q_sim), sim_filter(fn_sim), 'b-')
    ax2.loglog(sim_filter(q_sim), sim_filter(cross_sim), 'r-')
    ax2.loglog(sim_filter(q_sim), sim_filter(fm_sim), 'g-')
    q, fn, cross, fm = unpol_analysis(
        detector_8_up,
        detector_8_down,
        intensity_getter= lambda d: d.intensity
    )
    exp_filter = lambda arr : arr[10:-6:4]
    ax2.loglog(exp_filter(q), exp_filter(fn), 'b.')#, label = 'F$_{N}^2$')
    ax2.loglog(exp_filter(q), exp_filter(cross), 'r.')#, label = 'F$_{N}$F$_{M}$')
    ax2.loglog(exp_filter(q), exp_filter(fm), 'g.')#, label = 'F$_{M}^2$')

    q_sim, fn_sim, cross_sim, fm_sim = unpol_analysis(
        detector_8_up,
        detector_8_down,
        intensity_getter= lambda d: d.simulated_intensity
    )
    sim_filter = lambda arr: arr[10:-6]
    ax2.loglog(sim_filter(q_sim), sim_filter(fn_sim), 'b-')
    ax2.loglog(sim_filter(q_sim), sim_filter(cross_sim), 'r-')
    ax2.loglog(sim_filter(q_sim), sim_filter(fm_sim), 'g-')
    q, fn, cross, fm = unpol_analysis(
        detector_2_up,
        detector_2_down,
        intensity_getter= lambda d: d.intensity
    )
    exp_filter = lambda arr : arr[10:-15:4]
    ax2.loglog(exp_filter(q), exp_filter(fn), 'b.')#, label = 'F$_{N}^2$')
    ax2.loglog(exp_filter(q), exp_filter(cross), 'r.')#, label = 'F$_{N}$F$_{M}$')
    ax2.loglog(exp_filter(q), exp_filter(fm), 'g.')#, label = 'F$_{M}^2$')

    q_sim, fn_sim, cross_sim, fm_sim = unpol_analysis(
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
    ax3.text(0.05, 0.92, r'(c)', color = "black", fontsize = FONT_SIZE,horizontalalignment='left', verticalalignment='center', transform=ax3.transAxes)
    
    ax3.set_box_aspect(1)

    fig.tight_layout()

    fig.savefig(file_maker("magnetization_fig", ".eps"))
    fig.savefig(file_maker("magnetization_fig", ".pdf"))

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
    figure_form_factors()
    figure_particle_maps()
    figure_algorithm_performance()
    figure_polarization()
    figure_polarization_v2()
    #quick_test()

if __name__ == "__main__":
    main()

#%%
