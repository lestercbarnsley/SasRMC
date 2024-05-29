#%%
from dataclasses import dataclass, field
from functools import wraps
import time

from sas_rmc.evaluator import Evaluator
from sas_rmc.logger import LogCallback, NoLogCallback
from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.controller import Controller




def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
    
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        
        print(f"{my_func.__name__} took {(tend - tstart)} seconds to execute")
        return output
    return timed



@dataclass
class Simulator:
    controller: Controller
    state: ScatteringSimulation
    evaluator: Evaluator
    log_callback: LogCallback = field(default_factory=NoLogCallback)

    def start(self) -> None:
        starting_state = self.state
        self.log_callback.start(starting_state.get_loggable_data() | self.evaluator.get_loggable_data())

    def __enter__(self):
        self.start()
        return self

    def simulate(self) -> None:
        for command, acceptance_scheme in self.controller.ledger:
            new_state = command.execute(self.state)
            command_document = command.get_document()
            evaluation = self.evaluator.evaluate(new_state, acceptance_scheme)
            evaluation_document = self.evaluator.get_document()
            if evaluation:
                self.state = new_state
            self.log_callback.event(command_document | evaluation_document)

    def stop(self) -> None:
        ending_stage = self.state
        self.log_callback.stop(ending_stage.get_loggable_data() | self.evaluator.get_loggable_data())
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.stop()


        
if __name__ == "__main__":
    pass

   #%%