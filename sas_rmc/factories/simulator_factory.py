
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
    def create_simulator(self, controller: Controller, simulation: ScatteringSimulation, box_list: List[Box]) -> Simulator:
        pass


@dataclass
class MemorizedSimulatorFactory(SimulatorFactory):

    def create_simulator(self, controller: Controller, simulation: ScatteringSimulation, box_list: List[Box]) -> Simulator:
         return MemorizedSimulator(
            controller=controller,
            evaluator=MonteCarloEvaluator(
                simulation=simulation,
                viewer=CLIViewer()
            ),
            simulation=simulation,
            box_list=box_list
        )



