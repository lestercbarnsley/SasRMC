#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .simulator import Simulator
from .logger import Logger


@dataclass
class Runner(ABC):

    @abstractmethod
    def run(self) -> None:
        pass


@dataclass
class RmcRunner(Runner):
    logger: Logger
    simulator: Simulator
    force_log: bool = True


    def run_force_log(self) -> None:
        with self.logger:
            self.simulator.simulate()

    def run_not_forced_log(self) -> None:
        self.logger.before_event()
        self.simulator.simulate()
        self.logger.after_event()

    def run(self) -> None:
        if self.force_log:
            self.run_force_log()
        else:
            self.run_not_forced_log()



if __name__ == "__main__":
    pass
