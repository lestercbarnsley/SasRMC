#%%
from dataclasses import dataclass, field

from typing_extensions import Self
from pytest import mark, fixture
import numpy as np

from sas_rmc.box_simulation import Box, Cube
from sas_rmc.particles.particle import Particle
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam, SimulationConstant
from sas_rmc.vector import Vector
from sas_rmc.particles import ParticleResult, SphericalParticle


@dataclass
class MockParticleResult(ParticleResult):
    particle: Particle = field(
        default_factory=lambda : SphericalParticle.gen_from_parameters(
            position=Vector.null_vector(),
            sphere_radius=100
        )
    )

    def get_particle(self) -> Particle:
        return self.particle
    
    def change_particle(self, particle: Particle) -> Self:
        return type(self)(particle = particle)
    

@fixture
def simulation_state() -> ScatteringSimulation:
    return ScatteringSimulation(
        scale_factor=SimulationParam(1, 'test', bounds=(0, np.inf)),
        box_list=[Box(
            particle_results=[MockParticleResult()],
            cube=Cube(Vector.null_vector(), Vector(0, 1, 0), 600_000, 600_000, 600_000))]
    )
    

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
    assert simulation_state.get_physical_acceptance_strong() # The other tests cover the other cases

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



