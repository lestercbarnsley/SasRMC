from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass as pydantic_dataclass
import pandas as pd

from sas_rmc.controller import Controller
from sas_rmc.evaluator import Evaluator
from sas_rmc.loggers.logger import LogCallback
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.simulator import Simulator


@pydantic_dataclass
class SimulationStateFactory(ABC):
    @abstractmethod
    def create_simulation_state(self) -> ScatteringSimulation:
        pass

    @classmethod
    @abstractmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        pass






@pydantic_dataclass
class LoggerFactory(ABC):
    @abstractmethod
    def create_callbacks(self) -> LogCallback:
        pass

    @classmethod
    @abstractmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        pass



@pydantic_dataclass
class RunnerFactory:
    controller_factory: ControllerFactory
    state_factory: SimulationStateFactory
    evaluator_factory: EvaluatorFactory
    callback_factory: LoggerFactory
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

    @classmethod
    @abstractmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        pass


