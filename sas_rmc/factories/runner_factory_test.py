

from pathlib import Path
from pydantic.dataclasses import dataclass as pydantic_dataclass
import pandas as pd

from sas_rmc.factories import parse_data
from sas_rmc.factories.logger_factory import LoggerFactory, create_logger_from_dataframes
from sas_rmc.rmc_runner import RmcRunner
from sas_rmc.simulator import Simulator
from sas_rmc.factories.controller_factory import ControllerFactory
from sas_rmc.factories.evaluator_factory import EvaluatorFactory, create_evaluator_factory_from
from sas_rmc.factories.simulation_factory import SimulationStateFactory


@pydantic_dataclass
class RunnerFactory:
    controller_factory: ControllerFactory
    state_factory: SimulationStateFactory
    evaluator_factory: EvaluatorFactory
    callback_factory: LoggerFactory
    force_log: bool

    def create_runner(self) -> RmcRunner:
        evaluator=self.evaluator_factory.create_evaluator()
        state = self.state_factory.create_simulation_state(self.evaluator_factory.default_box_dimensions())
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
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame], results_folder: Path):
        state_factory = SimulationStateFactory.create_from_dataframes(dataframes)
        evaluator_factory = create_evaluator_factory_from(dataframes)
        controller_factory = ControllerFactory.create_from_dataframes(dataframes)
        logger_factory = create_logger_from_dataframes(dataframes, results_folder)
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return RunnerFactory(
            controller_factory=controller_factory,
            state_factory=state_factory,
            evaluator_factory=evaluator_factory,
            callback_factory=logger_factory,
            **value_frame
        )


