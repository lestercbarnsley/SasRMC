#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sas_rmc.acceptance_scheme import AcceptanceScheme
from sas_rmc.scattering_simulation import ScatteringSimulation


@dataclass
class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, simulation_state: ScatteringSimulation, acceptance_scheme: AcceptanceScheme | None = None) -> bool:
        pass

    @abstractmethod
    def get_document(self) -> dict:
        pass

    @abstractmethod
    def get_loggable_data(self) -> dict:
        pass

# %%
