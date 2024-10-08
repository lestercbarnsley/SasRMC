#%%
from dataclasses import dataclass, field

from sas_rmc.evaluator import Evaluator
from sas_rmc.logger import LogCallback, NoLogCallback
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.controller import Controller


@dataclass
class Simulator:
    controller: Controller
    state: ScatteringSimulation
    evaluator: Evaluator
    log_callback: LogCallback = field(default_factory=NoLogCallback)

    def start(self) -> None:
        starting_state = self.state
        self.log_callback.start(starting_state.get_loggable_data() | self.evaluator.get_loggable_data(starting_state))

    def __enter__(self):
        self.start()
        return self

    def simulate(self) -> None:
        for command, acceptance_scheme in self.controller.ledger:
            new_state, command_document = command.execute_and_get_document(self.state)
            evaluation, evaluation_document = self.evaluator.evaluate_and_get_document(new_state, acceptance_scheme)
            if evaluation:
                self.state = new_state
            self.log_callback.event(command_document | evaluation_document)

    def stop(self) -> None:
        ending_state = self.state
        self.log_callback.stop(ending_state.get_loggable_data() | self.evaluator.get_loggable_data(ending_state))
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.stop()


        
if __name__ == "__main__":
    pass

   #%%