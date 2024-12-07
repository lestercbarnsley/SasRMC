from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass as pydantic_dataclass
import pandas as pd

from sas_rmc.loggers.logger import LogCallback
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.simulator import Simulator
from sas_rmc.factories.controller_factory import ControllerFactory
from sas_rmc.factories.evaluator_factory import EvaluatorFactory
from sas_rmc.factories.simulation_factory import SimulationStateFactory






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
        evaluator=self.evaluator_factory.create_evaluator()
        state = self.state_factory.create_simulation_state(evaluator.default_box_dimensions())
        return RmcRunner(
            simulator=Simulator(
                controller = self.controller_factory.create_controller(state),
                state = state,
                evaluator=evaluator,
                log_callback=self.callback_factory.create_callbacks()
            ),
            force_log=self.force_log
        )

    @classmethod
    @abstractmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        pass


