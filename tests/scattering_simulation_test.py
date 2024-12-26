#%%
from pytest import mark, fixture
import numpy as np

from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam, SimulationConstant
from sas_rmc.vector import Vector
from sas_rmc.factories import simulation_factory, particle_factory

@fixture
def simulation_state():
    return simulation_factory.SimulationStateFactory(
        particle_factory=particle_factory.CoreShellParticleFactory(
            profile_type=particle_factory.ProfileType.DETECTOR_IMAGE,
            core_magnetization=100_000,
            core_radius=100.0,
            core_polydispersity=0.1,
            shell_thickness=15.0,
            shell_polydispersity=0.0,
            core_sld=7,
            shell_sld=0,
            solvent_sld=1),
        box_factory=simulation_factory.BoxFactory(100, 0.004, 1),
        box_dimension_1=600_000,
        box_dimension_2=600_000,
        box_dimension_3=600_000
    ).create_simulation_state([600_000, 600_000, 600_000])
    

@mark.parametrize('value', [
    0,
    5,
    10.3
])
def test_set_value(value: float):
    s = SimulationParam(3, 'test')
    s_new = s.set_value(value)
    assert s_new.value == value

@mark.parametrize(
        ['value', 'bounds', 'acceptance'],
        [
            (3, (0, np.inf), True),
            (-3, (0, np.inf), False),
            (100, (-np.inf, np.inf), True)
        ]
)
def test_param_acceptance(value: float, bounds: tuple[float, float], acceptance: bool):
    s = SimulationParam(value, 'test', bounds)
    assert s.get_physical_acceptance() == acceptance

def test_has_loggable_data():
    assert isinstance(SimulationParam(3, 'test').get_loggable_data(), dict)
    assert isinstance(SimulationConstant(3, 'test').get_loggable_data(), dict)

@mark.parametrize(
        ['initial_value', 'changed_value'],
        [
            (3,4),
            (0,0),
            (1,1230.53)
        ]
)
def test_simulation_constant(initial_value: float, changed_value: float):
    s = SimulationConstant(initial_value, 'test')
    s_new = s.set_value(changed_value)
    assert s_new.value == s.value
    assert s.get_physical_acceptance()

def test_physical_acceptance_strong(simulation_state: ScatteringSimulation):
    assert simulation_state.get_physical_acceptance_strong()
    simulation_state_2 = simulation_state.set_scale_factor(-1)
    assert simulation_state_2.get_physical_acceptance_strong() == False
    new_particle = simulation_state.get_particle(0, 0).change_position(Vector(6_000_000, 0, 0))
    simulation_state_3 = simulation_state.change_particle(0, 0, new_particle)
    assert simulation_state_3.get_physical_acceptance_strong() == False

def test_set_scale_factor(simulation_state: ScatteringSimulation):
    simulation_state_2 = simulation_state.set_scale_factor(-1).validate_state()
    simulation_state_3 = simulation_state_2.set_scale_factor(+1)
    assert simulation_state_3.get_physical_acceptance()
    assert simulation_state_3.get_physical_acceptance_strong()

def test_change_particle(simulation_state: ScatteringSimulation):
    previous_particle = simulation_state.get_particle(0, 0)
    new_particle = previous_particle.change_position(Vector(6_000_000, 0, 0))
    simulation_state_2 = simulation_state.change_particle(0, 0, new_particle).validate_state()
    simulation_state_3 = simulation_state_2.change_particle(0, 0, previous_particle)
    assert simulation_state_3.get_physical_acceptance()
    assert simulation_state_3.get_physical_acceptance_strong()

def test_simulation_state_has_loggable_data(simulation_state: ScatteringSimulation):
    assert isinstance(simulation_state.get_loggable_data(), dict)



