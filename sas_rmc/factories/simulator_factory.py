
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ..controller import Controller
from ..simulator import Simulator, MemorizedSimulator, MonteCarloEvaluator
from ..viewer import CLIViewer
from ..scattering_simulation import ScatteringSimulation
from ..box_simulation import Box


@dataclass
class SimulatorFactory(ABC):

    @abstractmethod
    def create_simulator(self) -> Simulator:
        pass


@dataclass
class MemorizedSimulatorFactory(SimulatorFactory):
    controller: Controller
    simulation: ScatteringSimulation
    box_list: List[Box]
    
    def create_simulator(self) -> Simulator:
         return MemorizedSimulator(
            controller=self.controller,
            evaluator=MonteCarloEvaluator(
                simulation=self.simulation,
                viewer=CLIViewer()
            ),
            simulation=self.simulation,
            box_list=self.box_list
        )



