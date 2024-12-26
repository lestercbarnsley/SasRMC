#%%

from dataclasses import dataclass

from pytest import fixture, raises

from sas_rmc.acceptance_scheme import AcceptanceScheme, MetropolisAcceptance
from sas_rmc.controller import ControlStep
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.commands import Command
from sas_rmc.simulator import Simulator, Evaluator, Controller
from sas_rmc.loggers import NoLogCallback

from tests.scattering_simulation_test import simulation_state


@dataclass
class MockEvaluator(Evaluator):

    def evaluate_and_get_document(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme) -> tuple[bool, dict]:
        return True, {}
    
    def get_loggable_data(self, simulation_state: ScatteringSimulation) -> dict:
        return {}
    

@dataclass
class MockCommand(Command):

    def execute(self, scattering_simulation: ScatteringSimulation) -> ScatteringSimulation:
        return scattering_simulation

    def execute_and_get_document(self, scattering_simulation: ScatteringSimulation) -> tuple[ScatteringSimulation, dict]:
        return self.execute(scattering_simulation), {}
    

@fixture
def simulator(simulation_state: ScatteringSimulation) -> Simulator:
    step = ControlStep(command = MockCommand(), acceptance_scheme=MetropolisAcceptance(0))
    return Simulator(
        controller=Controller([step]),
        state=simulation_state,
        evaluator=MockEvaluator(),
        log_callback=NoLogCallback()
    )

def test_full_simulation(simulator: Simulator):
    with simulator as s:
        s.simulate()

def test_start_failure(simulator: Simulator):
    old_state = simulator.state
    new_state = old_state.set_scale_factor(-1)
    simulator.set_state(new_state)
    with raises(AssertionError):
        simulator.start()

def test_stop_failure(simulator: Simulator):
    old_state = simulator.state
    new_state = old_state.set_scale_factor(-1)
    simulator.set_state(new_state)
    with raises(AssertionError):
        simulator.stop()

