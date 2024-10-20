#%%
from dataclasses import dataclass

from typing_extensions import Self

from sas_rmc.evaluator import Evaluator
from sas_rmc.controller import Controller
from sas_rmc.loggers import LogCallback
from sas_rmc.scattering_simulation import ScatteringSimulation


@dataclass
class Simulator:
    controller: Controller
    state: ScatteringSimulation
    evaluator: Evaluator
    log_callback: LogCallback | None = None

    def start(self) -> None:
        starting_state = self.state
        if self.log_callback is not None:
            self.log_callback.start(starting_state.get_loggable_data() | self.evaluator.get_loggable_data(starting_state))

    def __enter__(self) -> Self:
        self.start()
        return self

    def simulate(self) -> None:
        for step in self.controller.ledger:
            command = step.command
            new_state, command_document = command.execute_and_get_document(self.state)
            evaluation, evaluation_document = self.evaluator.evaluate_and_get_document(new_state, step.acceptance_scheme)
            if evaluation:
                self.state = new_state
            if self.log_callback is not None:
                self.log_callback.event(command_document | evaluation_document)

    def stop(self) -> None:
        ending_state = self.state
        if self.log_callback is not None:
            self.log_callback.stop(ending_state.get_loggable_data() | self.evaluator.get_loggable_data(ending_state))
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.stop()


        
if __name__ == "__main__":
    pass

   #%%