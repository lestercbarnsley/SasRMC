#%%
import pytest

from typing import List

import numpy as np

from sas_rmc.box_simulation import Box
from sas_rmc.controller import Controller
from sas_rmc.particle import CoreShellParticle, Dumbbell
from sas_rmc.scattering_simulation import Fitter2D, ScatteringSimulation
from sas_rmc import Vector, SimulatedDetectorImage, Polarization, DetectorConfig, commands, shapes


PI = np.pi

def default_cube() -> shapes.Cube:
    return shapes.Cube(dimension_0=10000, dimension_1=10000, dimension_2=10000)

def default_particles(particle_number = 20):
    radius = 100
    return [CoreShellParticle.gen_from_parameters(
        position = Vector(0, 2 * i * radius, 0),
        core_radius=radius,
        thickness = 10,
        core_sld = 6
    ) for i in range(particle_number)]

def default_box(particle_number = 20) -> Box:
    box = Box(
        particles = default_particles(particle_number),
        cube = default_cube()
    )
    box.force_inside_box()
    return box

def default_dumbbell_particles(particle_number = 20):
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

def default_dumbbell_box(particle_number = 20):
    return Box(
        particles = default_dumbbell_particles(particle_number),
        cube = default_cube()
    )


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

def default_simulation(box_list: List[Box]) -> ScatteringSimulation:
    for box in box_list:
        box.force_inside_box()
    detector = default_detector_image()
    qx, qy = detector.qX, detector.qY
    fitter = Fitter2D.generate_standard_fitter(
        simulated_detectors=[detector],
        box_list=box_list,
        qx_array=qx,
        qy_array=qy
    )
    return ScatteringSimulation(fitter=fitter)

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
                    [p.orientation.z  for p in box]
                ) == 0

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
    box = default_dumbbell_box()
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
            for p in [particle_2.particle_1, particle_2.particle_2]:
                for p_3 in [particle_3.particle_1, particle_3.particle_2]:
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
    detector_image = default_detector_image()
    qx_array, qy_array = detector_image.qX, detector_image.qY
    d1 = box[13].modulated_form_array(qx_array, qy_array)
    command.execute()
    d2 = box[13].modulated_form_array(qx_array, qy_array)
    assert np.average(d2 - d1) == 0
    assert id(d2) == id(d1)

def test_modulated_form_cache_polish():
    box = default_box()
    box.force_inside_box()
    position = Vector(2144.6643,1231.12315,1532.123)
    position_2 = box[14].position
    command = commands.MoveParticleTo(box, 14, position)
    command2 = commands.MoveParticleTo(box, 14, position_2)
    detector_image = default_detector_image()
    qx_array, qy_array = detector_image.qX, detector_image.qY
    d1 = box[14].modulated_form_array(qx_array, qy_array)
    f1 = box[14].form_array(qx_array, qy_array)
    command.execute()
    d2 = box[14].modulated_form_array(qx_array, qy_array)
    f2 = box[14].form_array(qx_array, qy_array)
    command2.execute()
    d3 = box[14].modulated_form_array(qx_array, qy_array)
    f3 = box[14].form_array(qx_array, qy_array)
    assert np.average(d3 - d1) == 0
    assert id(d3) == id(d1)
    assert np.average(d3 - d2) != 0
    assert id(d3) != id(d2)
    assert np.average(f2 - f1) == 0
    assert np.average(f3 - f1) == 0
    assert id(f1) == id(f2)
    assert id(f1) == id(f3)

def test_dumbbell_modulated_and_form_array():
    box = default_dumbbell_box()
    for _ in range(50):
        box.force_inside_box()
        position = Vector(2144.6643,1231.12315,1532.123)
        position_2 = box[14].position
        position_d2 = box[14].particle_2.position
        command = commands.MoveParticleTo(box, 14, position)
        command2 = commands.MoveParticleTo(box, 14, position_2)
        detector_image = default_detector_image()
        qx_array, qy_array = detector_image.qX, detector_image.qY
        d1 = box[14].modulated_form_array(qx_array, qy_array)
        f1 = box[14].form_array(qx_array, qy_array)
        dcore_01 = box[14].particle_1.modulated_form_array(qx_array, qy_array)
        dcore_02 = box[14].particle_2.modulated_form_array(qx_array, qy_array)
        command.execute()
        d2 = box[14].modulated_form_array(qx_array, qy_array)
        f2 = box[14].form_array(qx_array, qy_array)
        dcore_11 = box[14].particle_1.modulated_form_array(qx_array, qy_array)
        dcore_12 = box[14].particle_2.modulated_form_array(qx_array, qy_array)
        command2.execute()
        d3 = box[14].modulated_form_array(qx_array, qy_array)
        f3 = box[14].form_array(qx_array, qy_array)
        position_d22 = box[14].particle_2.position
        dcore_21 = box[14].particle_1.modulated_form_array(qx_array, qy_array)
        dcore_22 = box[14].particle_2.modulated_form_array(qx_array, qy_array)
        assert box[14].position == position_2
        assert (position_d2 - position_d22).mag == pytest.approx(0)
        assert np.average(d3 - d1) == pytest.approx(0)
        assert np.average(dcore_01 - dcore_21) == pytest.approx(0)
        assert np.average(dcore_02 - dcore_22) == pytest.approx(0)
        assert id(dcore_01) == id(dcore_21)
        assert id(dcore_02) == id(dcore_22)
        assert np.average(d3 - d2) != pytest.approx(0)
        assert np.average(dcore_11 - dcore_21) != pytest.approx(0)
        assert np.average(dcore_12 - dcore_22) != pytest.approx(0)
        assert id(dcore_11) != id(dcore_21)
        assert id(dcore_12) != id(dcore_22)

def test_modulated_and_form_array():
    box = default_box()
    box.force_inside_box()
    detector_image = default_detector_image()
    qx_array, qy_array = detector_image.qX, detector_image.qY
    ff = box[13].form_array(qx_array, qy_array)
    fm = box[13].modulated_form_array(qx_array, qy_array)
    fm2 = ff * np.exp(1j * (qx_array * box[13].position.x + qy_array * box[13].position.y))
    assert np.average(fm2 - fm) == 0

def test_box_intensity():
    from sas_rmc.form_calculator import box_intensity
    box = default_box()
    box.force_inside_box()
    detector_image = default_detector_image()
    qx_array, qy_array = detector_image.qX, detector_image.qY
    b_tensity = box_intensity(box, qx_array, qy_array)
    get_mfas = lambda : [p.modulated_form_array(qx_array, qy_array) for p in box.particles]
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

    b_tensity_after = box_intensity(box, qx_array, qy_array)
    assert b_tensity.shape == b_tensity_after.shape
    assert np.average(b_tensity_after - b_tensity) != 0

def test_reset_form_cache():
    particles = default_particles()
    particle = particles[19]
    detector = default_detector_image()
    qx, qy = detector.qX, detector.qY
    particle.modulated_form_array(qx, qy)
    assert len(particle.form_cache._form_array) != 0
    assert len(particle.form_cache._modulated_form_array) != 0
    particle.form_cache.reset_form()
    assert len(particle.form_cache._form_array) == 0
    assert len(particle.form_cache._modulated_form_array) == 0
    

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
    metropolis = commands.MetropolisAcceptance(
        temperature=10,
        rng = 0.1,
        delta_chi=10,
        _after_chi = 5,
    )
    metropolis._acceptance_state = commands.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == True
    metropolis.rng = 0.4
    metropolis._acceptance_state = commands.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == False
    metropolis.temperature = 5
    metropolis._acceptance_state = commands.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == False
    metropolis.delta_chi = -0.000001
    metropolis._acceptance_state = commands.AcceptanceState.UNTESTED
    metropolis._calculate_success()
    assert metropolis.is_acceptable() == True

def test_metropolis_handle_simulation():
    box_list = [default_box()]
    simulation = default_simulation(box_list=box_list)
    metropolis = commands.MetropolisAcceptance(
        temperature=10,
        rng = 0.1,
        _after_chi = 5,
    )
    metropolis.handle_simulation(simulation)
    metropolis.set_physical_acceptance(simulation.get_physical_acceptance())
    assert metropolis.is_acceptable()
    assert metropolis.delta_chi < 0
    assert metropolis._after_chi == simulation.current_goodness_of_fit
    assert type(metropolis._after_chi) in [float, np.float64]

def test_simulation_get_acceptance():
    box_list = [default_box()]
    simulation = default_simulation(box_list=box_list)
    simulation.rescale_factor = 1.0
    simulation.magnetic_rescale = 1.0
    assert simulation.get_physical_acceptance()
    simulation.rescale_factor = -0.0000001
    assert simulation.get_physical_acceptance() == False
    simulation.rescale_factor = 1.0
    simulation.magnetic_rescale = -0.0000001
    assert simulation.get_physical_acceptance() == False
    simulation.rescale_factor = 1.0
    simulation.magnetic_rescale = 1.0
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
        acceptance_scheme=commands.MetropolisAcceptance(
            delta_chi=-1
        )
    ) for command in commands_for_controller]
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
    simulation = default_simulation(box_list=box_list)
    box = box_list[0]
    positions = [
        Vector(0, 50, 100),
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
        acceptance_scheme=commands.MetropolisAcceptance(
            delta_chi=-1
        )
    ) for command in commands_for_controller]
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
        distance_calc = lambda : dumbbell.particle_1.position.distance_from_vector(dumbbell.particle_2.position)
        orientation_calc = lambda : dumbbell.orientation
        old_distance = distance_calc()
        old_orientation = orientation_calc()
        dumbbell.position = dumbbell.position + Vector(50,50,50)
        assert old_distance == pytest.approx(distance_calc())
        assert old_orientation == orientation_calc()
        new_orientation = Vector(np.random.rand(), np.random.rand(), np.random.rand())
        dumbbell.orientation = new_orientation
        assert old_distance == pytest.approx(distance_calc())
        assert (new_orientation.unit_vector - orientation_calc()).mag == pytest.approx(0)

def test_set_particle_state():
    for _ in range(50):
        box = default_box()
        box.force_inside_box()
        position = box[4].position
        orientation = box[4].orientation
        magnetization = box[4].magnetization
        random_vec = lambda : Vector(np.random.rand(), np.random.rand(), np.random.rand())
        command_1 = commands.SetParticleState(box, 4, position=position, orientation=orientation, magnetization=magnetization)
        command_2 = commands.SetParticleState(box, 4, random_vec(), random_vec(), random_vec())
        command_2.execute()
        assert box[4].position != position
        assert box[4].orientation != orientation
        assert box[4].magnetization != magnetization
        command_1.execute()
        assert box[4].position == position
        assert box[4].orientation == orientation
        assert box[4].magnetization == magnetization

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
        undo_command = commands.SetParticleState.gen_from_particle_command(command)
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
            simulation=simulation,
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