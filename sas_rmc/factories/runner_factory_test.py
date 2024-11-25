from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc.controller import Controller
from sas_rmc.particles import Particle
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.simulator import Simulator



class ParticleFactory(ABC):
    @abstractmethod
    def create_particle(self) -> Particle:
        pass

class SimulationStateFactory(ABC):
    @abstractmethod
    def create_simulation_state(self) -> ScatteringSimulation:
        pass


class ControllerFactory(ABC):
    @abstractmethod
    def create_controller(self, simulation_state: ScatteringSimulation) -> Controller:
        pass






@pydantic_dataclass
class RunnerFactory:
    controller_factory: ControllerFactory
    state_factory: SimulationStateFactory
    force_log: bool


    def create_runner(self) -> RmcRunner:
        state = self.state_factory.create_simulation_state()
        return RmcRunner(
            simulator=Simulator(
                controller = self.controller_factory.create_controller(state),
                state = state,
                evaluator=self.evaluator_factory.create_evaluator(),
                log_callback=self.callback_factory.create_callbacks()
            ),
            force_log=self.force_log
        )



