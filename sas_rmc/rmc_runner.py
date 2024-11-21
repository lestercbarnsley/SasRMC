#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sas_rmc.simulator import Simulator


@dataclass
class Runner(ABC):

    @abstractmethod
    def run(self) -> None:
        pass


@dataclass
class RmcRunner(Runner):
    simulator: Simulator
    force_log: bool = True

    def run_force_log(self) -> None:
        with self.simulator as simulator:
            simulator.simulate()

    def run_not_forced_log(self) -> None:
        self.simulator.start()
        self.simulator.simulate()
        self.simulator.stop()

    def run(self) -> None:
        if self.force_log:
            self.run_force_log()
        else:
            self.run_not_forced_log()



if __name__ == "__main__":
    pass
