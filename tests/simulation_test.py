#%%
import pytest
from pathlib import Path

from typing import List

import numpy as np
import sas_rmc

from sas_rmc.box_simulation import Box
from sas_rmc.controller import Controller
from sas_rmc.particle import CoreShellParticle, Dumbbell, Particle
from sas_rmc.scattering_simulation import MAGNETIC_RESCALE, NUCLEAR_RESCALE, ScatteringSimulation
from sas_rmc import Vector, SimulatedDetectorImage, Polarization, DetectorConfig, commands, shapes
from sas_rmc.fitter import Fitter2D


PI = np.pi

def default_cube() -> shapes.Cube:
    return shapes.Cube(dimension_0=10000, dimension_1=10000, dimension_2=10000)

def default_particles(particle_number = 20) -> List[CoreShellParticle]:
    radius = 100
    return [CoreShellParticle.gen_from_parameters(
        position = Vector(0, 0, 0),
        core_radius=radius,
        thickness = 10,
        core_sld = 6
    ) for _ in range(particle_number)]

def default_box(particles: List[Particle] = None) -> Box:
    if particles is None:
        particles = default_particles()
    box = Box(
        particles = particles,
        cube = default_cube()
    )
    box.force_inside_box()
    return box

def structure_box(box: Box) -> None:
    radius =  np.average([p.shapes[1].radius for p in box.particles])
    for i, particle in enumerate(box.particles):
        box.particles[i] = particle.set_position(Vector(0, 2.2 * i * radius))



def default_dumbbell_particles(particle_number = 20) -> List[Dumbbell]:
    p1_radius = 30
    p2_radius = 50
    return [Dumbbell.gen_from_parameters(
        core_radius=p1_radius,
        seed_radius=p2_radius,
        shell_thickness = 10,
        core_sld=4,
        seed_sld =6,
        shell_sld = 1,
        solvent_sld = 0,
        ) for _ in range(particle_number)]


def default_detector_image() -> SimulatedDetectorImage:
    file = Path.cwd() / Path('data') / Path('I-unpolarized_test_data_from_KWS-1.DAT')
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
    detector_image = SimulatedDetectorImage.gen_from_txt(file, DETECTOR_CONFIG_C14D14)
    return detector_image

DEFAULT_DETECTOR_IMAGE = default_detector_image()

def default_simulation(box_list: List[Box]) -> ScatteringSimulation:
    for box in box_list:
        box.force_inside_box()
    detector = default_detector_image()
    qx, qy = np.meshgrid(sas_rmc.detector.average_uniques(detector.qX), sas_rmc.detector.average_uniques(detector.qY))
    fitter = Fitter2D.generate_standard_fitter(
        detector_list=[detector],
        box_list = box_list,
        result_calculator_maker=lambda detector : sas_rmc.result_calculator.AnalyticalCalculator(qx, qy)
    )
    return ScatteringSimulation(fitter=fitter, simulation_params=sas_rmc.simulator_factory.box_simulation_params_factory())

def test_particle_move():
    box = default_box()
    for _ in range(50):
        box.force_inside_box()
        command = commands.MoveParticleBy(box, 11, Vector(11, 61, -13.34))
        command.execute()
        position = box[11].position
        for shape in box[11].shapes:
            assert shape.central_position == position
        #Dumbbell move and rotation tested in test_dumbbell_rotation()


def test_box_disperse():
    radii = [10,20,50, 100, 200]
    cube = shapes.Cube(dimension_0=10000, dimension_1=10000, dimension_2=10000)
    for radius in radii:
        particle_number = 50
        particles = [CoreShellParticle.gen_from_parameters(position = Vector.null_vector(), core_radius=radius, thickness = 10, core_sld=6) for _ in range(particle_number)]
        box = Box(particles = particles, cube = cube)
        for in_plane in [True, False]:
            box.force_inside_box(in_plane=in_plane)
            assert box.collision_test() == False
            if in_plane:
                assert all(
                    [p.position.z == 0 for p in box]
                )
                assert np.sum(
                    [p.orientation.z for p in box]
                ) == 0

def test_collision_detection():
    shape_list_1 = [shapes.Sphere(radius=100)]
    shape_list_2 = [shapes.Sphere(radius=100)]
    assert shapes.collision_detected(shape_list_1, shape_list_2) == True
    shape_list_2 = [shapes.Sphere(central_position=Vector(200, 200, 0))]
    assert shapes.collision_detected(shape_list_1, shape_list_2) == False
    shape_list_2 = [shapes.Sphere(central_position=Vector(100, 0,0))]
    assert shapes.collision_detected(shape_list_1, shape_list_2) == True

def test_particle_collision_detection():
    core_shell_particles = default_particles()
    assert core_shell_particles[0].collision_detected(core_shell_particles[1]) == True
    new_particle_1 = core_shell_particles[1].set_position(Vector(2000, 2000))
    assert core_shell_particles[0].collision_detected(new_particle_1) == False


def test_particle_jump():
    box = default_box()
    for _ in range(50):
        box.force_inside_box()
        command = commands.JumpParticleTo(box, 2, 3)
        command.execute()
        particle_position_2 = box[2].position
        particle_3_position = box[3].position
        assert particle_position_2.distance_from_vector(particle_3_position) == pytest.approx(box[2].shapes[1].radius + box[3].shapes[1].radius)
        assert box.collision_test() == False

def test_dumbbell_jump():
    box = default_box(default_dumbbell_particles())
    for in_plane in [True, False]:
        happened = []
        for _ in range(50):
            box.force_inside_box(in_plane = in_plane)
            assert box.collision_test() == False
            command = commands.JumpParticleTo(box, 2, 3)
            command.execute()
            particle_2 = box[2]
            particle_3 = box[3]
            separations, radii_sums = [], []
            for p in particle_2.particle_list:
                for p_3 in particle_3.particle_list:
                    separations.append(p.position.distance_from_vector(p_3.position))
                    radii_sums.append(p.shapes[1].radius + p_3.shapes[1].radius)
            if box.collision_test() == False:
                happened.append(True)
                assert any([s == pytest.approx(sum_rads) for s, sum_rads in zip(separations, radii_sums)])
                assert box.collision_test() == False
        assert sum(happened) > 25

def test_particle_orbit():
    box = default_box()
    for _ in range(50):
        box.force_inside_box()
        command = commands.OrbitParticle(box, 7, PI/8)
        particle_near = box.get_nearest_particle(box[7])
        distance_calculator = lambda : particle_near.position.distance_from_vector(box[7].position)
        distance_prior = distance_calculator()
        command.execute()
        distance_after = distance_calculator()
        assert distance_prior == pytest.approx(distance_after)

def test_small_move():
    box = default_box()
    for _ in range(50):
        box.force_inside_box()
        command = commands.MoveParticleBy(box, 15, Vector(20,20,20))
        position_prior = box[15].position
        command.execute()
        position_after = box[15].position
        assert (position_after - position_prior - Vector(20, 20, 20)).mag == pytest.approx(0)

def test_large_move():
    box = default_box()
    box.force_inside_box()
    position = Vector(2144.6643,1231.12315,1532.123)
    command = commands.MoveParticleTo(box, 14, position)
    command.execute()
    assert (box[14].position - position).mag == pytest.approx(0)

def test_modulated_form_cache():
    box = default_box()
    box.force_inside_box()
    
    position = Vector(2144.6643,1231.12315,1532.123)
    command = commands.MoveParticleTo(box, 14, position)
    detector_image = DEFAULT_DETECTOR_IMAGE
    qx_array, qy_array = detector_image.qX, detector_image.qY
    form_calculator = sas_rmc.result_calculator.AnalyticalCalculator(qx_array, qy_array)
    get_modulated_form_array = lambda : form_calculator.modulated_form_array(box[13], orientation=box[13].orientation, position=box[13].position)
    d1 = get_modulated_form_array()
    command.execute()
    d2 = get_modulated_form_array()
    assert np.average(d2 - d1) == 0
    assert id(d2) == id(d1)

def test_modulated_form_cache_polish():
    box = default_box()
    box.force_inside_box()
    position = Vector(2144.6643,1231.12315,1532.123)
    position_2 = box[14].position
    command = commands.MoveParticleTo(box, 14, position)
    command2 = commands.MoveParticleTo(box, 14, position_2)
    detector_image = DEFAULT_DETECTOR_IMAGE
    qx_array, qy_array = detector_image.qX, detector_image.qY
    form_calculator = sas_rmc.result_calculator.AnalyticalCalculator(qx_array, qy_array)
    get_modulated_form = lambda : form_calculator.modulated_form_array(box[14], orientation=box[14].orientation, position=box[14].position)
    d1 = get_modulated_form()
    f1 = box[14].form_array(qx_array, qy_array, orientation=box[14].orientation)
    command.execute()
    d2 = get_modulated_form()
    f2 = box[14].form_array(qx_array, qy_array, orientation=box[14].orientation)
    command2.execute()
    d3 = get_modulated_form()
    f3 = box[14].form_array(qx_array, qy_array, orientation=box[14].orientation)
    assert np.average(d3 - d1) == 0
    #assert id(d3) == id(d1) # This worked when particles were mutable, so I might delete this!
    assert np.average(d3 - d2) != 0
    assert id(d3) != id(d2)
    assert np.average(f2 - f1) == 0
    assert np.average(f3 - f1) == 0
    assert id(f1) != id(f2)
    assert id(f1) != id(f3)

def test_dumbbell_modulated_and_form_array():
    detector_image = DEFAULT_DETECTOR_IMAGE
    qx_array, qy_array = detector_image.qX, detector_image.qY
    form_calculator = sas_rmc.result_calculator.AnalyticalCalculator(qx_array, qy_array)
    get_current_form_array = lambda p: p.form_array(qx_array, qy_array, orientation = p.orientation)
    get_current_modulated_form_array = lambda p: form_calculator.modulated_form_array(p , position = p.position, orientation = p.orientation)
    for _ in range(50):
        box = default_box(default_dumbbell_particles())
        position = Vector(2144.6643,1231.12315,1532.123)
        position_2 = box[14].position
        position_d2 = box[14].seed_particle.position
        command = commands.MoveParticleTo(box, 14, position)
        command2 = commands.MoveParticleTo(box, 14, position_2)
        d1 = get_current_modulated_form_array(box[14])#.modulated_form_array(qx_array, qy_array, position=box[14].position, orientation=box[14].orientation)
        f1 = get_current_form_array(box[14])#.form_array(qx_array, qy_array, orientation=box[14].orientation)
        dcore_01 = get_current_modulated_form_array(box[14].core_particle)#.modulated_form_array(qx_array, qy_array, position=box[14].particle_1.position, orientation=box[14].particle_1.orientation)
        dcore_02 = get_current_modulated_form_array(box[14].seed_particle)#.modulated_form_array(qx_array, qy_array,position=box[14].particle_2.position, orientation=box[14].particle_2.orientation)
        command.execute()
        d2 = get_current_modulated_form_array(box[14])#box[14].modulated_form_array(qx_array, qy_array, position=box[14].position, orientation=box[14].orientation)
        f2 = get_current_form_array(box[14])#box[14].form_array(qx_array, qy_array, orientation=box[14].orientation)
        dcore_11 = get_current_modulated_form_array(box[14].core_particle)#box[14].particle_1.modulated_form_array(qx_array, qy_array, position=box[14].particle_1.position, orientation=box[14].particle_1.orientation)
        dcore_12 = get_current_modulated_form_array(box[14].seed_particle)#box[14].particle_2.modulated_form_array(qx_array, qy_array,position=box[14].particle_2.position, orientation=box[14].particle_2.orientation)
        command2.execute()
        d3 = get_current_modulated_form_array(box[14])#box[14].modulated_form_array(qx_array, qy_array, position=box[14].position, orientation=box[14].orientation)
        f3 = get_current_form_array(box[14])#box[14].form_array(qx_array, qy_array, orientation=box[14].orientation)
        position_d22 = box[14].seed_particle.position
        dcore_21 = get_current_modulated_form_array(box[14].core_particle)#box[14].particle_1.modulated_form_array(qx_array, qy_array, position=box[14].particle_1.position, orientation=box[14].particle_1.orientation)
        dcore_22 = get_current_modulated_form_array(box[14].seed_particle)#box[14].particle_2.modulated_form_array(qx_array, qy_array,position=box[14].particle_2.position, orientation=box[14].particle_2.orientation)
        assert box[14].position == position_2
        assert (position_d2 - position_d22).mag == pytest.approx(0)
        assert np.average(d3 - d1) == pytest.approx(0)
        assert np.average(dcore_01 - dcore_21) == pytest.approx(0)
        assert np.average(dcore_02 - dcore_22) == pytest.approx(0)
        #assert id(dcore_01) == id(dcore_21)
        #assert id(dcore_02) == id(dcore_22) This probably only passes if the particle is mutable
        assert np.average(d3 - d2) != pytest.approx(0)
        assert np.average(dcore_11 - dcore_21) != pytest.approx(0)
        assert np.average(dcore_12 - dcore_22) != pytest.approx(0)
        assert id(dcore_11) != id(dcore_21)
        assert id(dcore_12) != id(dcore_22)

def test_modulated_and_form_array():
    box = default_box()
    box.force_inside_box()
    detector_image = DEFAULT_DETECTOR_IMAGE
    qx_array, qy_array = detector_image.qX, detector_image.qY
    form_calculator = sas_rmc.result_calculator.AnalyticalCalculator(qx_array, qy_array)
    ff = box[13].form_array(qx_array, qy_array, orientation=box[13].orientation)
    fm = form_calculator.modulated_form_array( box[13], position=box[13].position, orientation=box[13].orientation)
    fm2 = ff * np.exp(1j * (qx_array * box[13].position.x + qy_array * box[13].position.y))
    assert np.average(fm2 - fm) == 0

def test_box_intensity():
    from sas_rmc.form_calculator import box_intensity
    box = default_box()
    box.force_inside_box()
    detector_image = DEFAULT_DETECTOR_IMAGE
    qx_array, qy_array = detector_image.qX, detector_image.qY
    form_calculator = sas_rmc.result_calculator.AnalyticalCalculator(qx_array, qy_array)
    b_tensity = box_intensity([form_calculator.form_result(p) for p in box.particles], box.volume, qx_array, qy_array)
    get_mfas = lambda : [form_calculator.form_result(p).form_nuclear for p in box.particles]
    mfas_prior = get_mfas()
    command = commands.MoveParticleBy(box, 15, Vector(20,20,20))
    command.execute()
    mfas_after = get_mfas()
    for index, (mfa_prior, mfa_after) in enumerate(zip(mfas_prior, mfas_after)):
        if index == 15:
            assert np.average(mfa_after - mfa_prior) != 0
            assert id(mfa_after) != id(mfa_prior)
        else:
            assert np.average(mfa_after - mfa_prior) == 0
            assert id(mfa_after) == id(mfa_prior)

    b_tensity_after = box_intensity([form_calculator.form_result(p) for p in box.particles], box.volume, qx_array, qy_array)
    assert b_tensity.shape == b_tensity_after.shape
    assert np.average(b_tensity_after - b_tensity) != 0

def test_controller():
    box = default_box()
    box.force_inside_box()
    positions = [
        Vector(0, 50, 100),
        Vector(15,15,15),
        Vector(-400, -400, 0),
    ]
    commands_for_controller = [
        commands.MoveParticleTo(box, 3, positions[0]),
        commands.MoveParticleBy(box, 3, positions[1]),
        commands.MoveParticleBy(box, 3, positions[2]),
    ]
    controller = Controller(commands_for_controller)
    controller.action()
    controller.compute_states()
    assert box[3].position == positions[0]
    assert controller._current == 1
    assert len(controller.completed_commands) == controller._current
    controller.action()
    controller.compute_states()
    assert box[3].position == positions[0] + positions[1]
    assert controller._current == 2
    assert len(controller.completed_commands) == controller._current
    controller.action()
    controller.compute_states()
    assert box[3].position == positions[0] + positions[1] + positions[2]
    assert controller._current == 3
    assert len(controller.completed_commands) == controller._current
    
def test_metropolis_acceptance():
    metropolis = sas_rmc.acceptance_scheme.MetropolisAcceptance(
        temperature=10,
        rng_val = 0.1,
    )
    metropolis.set_delta_chi(delta_chi=10, after_chi=5)
    metropolis._acceptance_state = sas_rmc.acceptance_scheme.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == True
    metropolis.rng_val = 0.4
    metropolis._acceptance_state = sas_rmc.acceptance_scheme.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == False
    metropolis.temperature = 5
    metropolis._acceptance_state = sas_rmc.acceptance_scheme.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == False
    metropolis.set_delta_chi(delta_chi=-0.000001, after_chi=5)
    metropolis._acceptance_state = sas_rmc.acceptance_scheme.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == True

def test_metropolis_handle_simulation():
    box_list = [default_box()]
    simulation = default_simulation(box_list=box_list)
    metropolis = sas_rmc.acceptance_scheme.MetropolisAcceptance(
        temperature=10,
        rng_val = 0.1,
    )
    metropolis.set_delta_chi(delta_chi=0, after_chi=5)
    metropolis.handle_simulation(simulation)
    metropolis.set_physical_acceptance(simulation.get_physical_acceptance())
    assert metropolis.is_acceptable()
    assert metropolis.delta_chi < 0
    assert metropolis._after_chi == simulation.current_goodness_of_fit
    assert type(metropolis._after_chi) in [float, np.float64]

def test_simulation_get_acceptance():
    box_list = [default_box()]
    simulation = default_simulation(box_list=box_list)
    simulation.simulation_params.set_value(NUCLEAR_RESCALE, 1.0)
    simulation.simulation_params.set_value(MAGNETIC_RESCALE, 1.0)
    assert simulation.get_physical_acceptance()
    simulation.simulation_params.set_value(NUCLEAR_RESCALE, -0.0000001)
    assert simulation.get_physical_acceptance() == False
    simulation.simulation_params.set_value(NUCLEAR_RESCALE, 1.0)
    simulation.simulation_params.set_value(MAGNETIC_RESCALE, -0.0000001)
    assert simulation.get_physical_acceptance() == False
    simulation.simulation_params.set_value(NUCLEAR_RESCALE, 1.0)
    simulation.simulation_params.set_value(MAGNETIC_RESCALE, 1.0)
    command = commands.MoveParticleTo(box_list[0], 13, Vector(1e10, 1e10, 1e10))
    command.execute()
    assert box_list[0].collision_test() != False
    box_list[0].force_inside_box()
    assert box_list[0].collision_test() != True
    
def test_acceptable_command():
    box = default_box()
    positions = [
        Vector(0, 50, 100),
        Vector(15,15,15),
        Vector(-400, -400, 0),
    ]
    commands_for_controller = [
        commands.MoveParticleTo(box, 3, positions[0]),
        commands.MoveParticleBy(box, 3, positions[1]),
        commands.MoveParticleBy(box, 3, positions[2]),
    ]
    decorated_commands = [commands.AcceptableCommand(
        base_command = command,
        acceptance_scheme=sas_rmc.acceptance_scheme.MetropolisAcceptance()
    ) for command in commands_for_controller]
    for c in decorated_commands:
        c.acceptance_scheme.set_delta_chi(-1, after_chi = 0)
    controller = Controller(ledger = decorated_commands)
    for command in controller.ledger:
        controller.action()
        controller.compute_states()
    assert box[3].position == positions[0] + positions[1] + positions[2]
    controller.ledger[2].acceptance_scheme.set_physical_acceptance(False)
    assert controller.ledger[2].acceptance_scheme.is_acceptable() == False
    controller.compute_states()
    assert box[3].position == positions[0] + positions[1]

def test_controller_acceptance_independence():
    box_list = [default_box()]
    #simulation = default_simulation(box_list=box_list)
    box = box_list[0]
    structure_box(box)
    positions = [
        Vector(0, 50, 500),
        Vector(15,15,15),
        Vector(-400, -400, 0),
        Vector(1e7, 1e7, 1e7),
        Vector(13, 13, 13)
    ]
    commands_for_controller = [
        commands.MoveParticleTo(box, 3, positions[0]),
        commands.MoveParticleBy(box, 3, positions[1]),
        commands.MoveParticleBy(box, 3, positions[2]),
        commands.MoveParticleBy(box, 3, positions[3]),
        commands.MoveParticleBy(box, 3, positions[4]),
        ]
    decorated_commands = [commands.AcceptableCommand(
        base_command = command,
        acceptance_scheme=sas_rmc.acceptance_scheme.MetropolisAcceptance()
    ) for command in commands_for_controller]
    for c in decorated_commands:
        c.acceptance_scheme.set_delta_chi(-1, after_chi = 0)
    controller = Controller(decorated_commands)
    for i, command in enumerate(controller.ledger):
        controller.action()
        controller.compute_states()
        unacceptable = box.collision_test()
        command.acceptance_scheme.set_physical_acceptance(not unacceptable)
        assert box[3].position == [
            positions[0],
            positions[0] + positions[1],
            positions[0] + positions[1] + positions[2],
            positions[0] + positions[1] + positions[2] + positions[3],
            positions[0] + positions[1] + positions[2] + positions[4],
            ][i]

def test_dumbbell_rotation():
    for _ in range(50):
        dumbbell = Dumbbell.gen_from_parameters(50, 50, 10, 4, 6, 1, 0)
        distance_calc = lambda db : db.particle_list[0].position.distance_from_vector(db.particle_list[1].position)
        orientation_calc = lambda db : db.orientation
        old_distance = distance_calc(dumbbell)
        old_orientation = orientation_calc(dumbbell)
        dumbbell = dumbbell.set_position( dumbbell.position + Vector(50,50,50) )
        assert old_distance == pytest.approx(distance_calc(dumbbell))
        assert old_orientation == orientation_calc(dumbbell)
        new_orientation = Vector(np.random.rand(), np.random.rand(), np.random.rand())
        dumbbell = dumbbell.set_orientation(new_orientation )
        assert old_distance == pytest.approx(distance_calc(dumbbell))
        assert (new_orientation.unit_vector - orientation_calc(dumbbell)).mag == pytest.approx(0)

def test_set_particle_state():
    for _ in range(50):
        box = default_box()
        box.force_inside_box()
        position = box[4].position
        orientation = box[4].orientation
        magnetization = box[4].magnetization
        random_vec = lambda : Vector(np.random.rand(), np.random.rand(), np.random.rand())
        command_1 = commands.SetParticleState(box, 4, box[4])
        command_2 = commands.SetParticleState(box, 4, box[4].set_magnetization(random_vec()).set_position(random_vec())
        )
        command_2.execute()
        assert box[4].position != position
        assert box[4].orientation == orientation # These are default particles, so orientation changes are not allowed
        assert box[4].magnetization != magnetization
        command_1.execute()
        assert box[4].position == position
        assert box[4].orientation == orientation
        assert box[4].magnetization == magnetization

def test_set_particle_magnetization():
    for _ in range(50):
        box = default_box()
        box.force_inside_box()
        get_particle = lambda i : box.particles[i]
        particle_4_old = get_particle(4)
        magnetization = particle_4_old.magnetization
        new_magnetization = Vector.random_vector(470e3)
        command_1 = commands.MagnetizeParticle(box, 4, new_magnetization)
        command_1.execute()
        particle_4_new = get_particle(4)
        assert particle_4_old != particle_4_new
        assert particle_4_new.magnetization == new_magnetization
        assert particle_4_old.magnetization == magnetization

def test_physical_acceptance_half_test():
    box = default_box()
    simulation = default_simulation(box_list=[box])
    for _ in range(100):
        particle_index = np.random.choice(range(len(box)))
        command = commands.MoveParticleBy(
            box = box,
            particle_index=particle_index,
            position_delta=Vector.random_vector_xy(100)
        )
        undo_command = commands.SetParticleState.gen_from_particle(box, particle_index)#.gen_from_particle_command(command)
        command.execute()
        weak_physical_acceptance = command.physical_acceptance_weak()
        assert weak_physical_acceptance != box.collision_test()
        if not weak_physical_acceptance:
            undo_command.execute()

        
def test_physical_acceptance_simulator_scale():
    box = default_box()
    simulation = default_simulation(box_list=[box])
    for _ in range(500):
        command = commands.NuclearRescale(
            simulation_params=simulation.simulation_params,
            change_by_factor=np.random.uniform(
                low = 0.1,
                high = 1.9,
            )
        )
        command.execute()
        assert simulation.get_physical_acceptance() == command.physical_acceptance_weak()




        
#box_disperse()
#%%

#%%