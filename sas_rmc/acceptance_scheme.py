
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from sas_rmc.scattering_simulation import ScatteringSimulation
from sas_rmc.constants import RNG

rng = RNG


class AcceptanceState(Enum):
    UNTESTED = "untested"
    ACCEPTABLE = "acceptable"
    UNACCEPTABLE = "unacceptable"


@dataclass
class AcceptanceScheme(ABC):
    _acceptance_state: AcceptanceState = field( default_factory = lambda : AcceptanceState.UNTESTED, init = False)

    @abstractmethod
    def handle_simulation(self, simulation: ScatteringSimulation) -> None:
        pass

    def set_physical_acceptance(self, physical_acceptance: bool) -> None:
        if not physical_acceptance:
            self._acceptance_state = AcceptanceState.UNACCEPTABLE

    @abstractmethod
    def is_acceptable(self) -> bool:
        return True

    def get_loggable_data(self) -> dict:
        return {}


@dataclass
class UnconditionalAcceptance(AcceptanceScheme):
    
    def handle_simulation(self, simulation: ScatteringSimulation) -> None:
        return super().handle_simulation(simulation)

    def is_acceptable(self) -> bool:
        return True


@dataclass
class MetropolisAcceptance(AcceptanceScheme):
    temperature: float = 0
    rng_val: float = field(default_factory= rng.uniform)
    delta_chi: float = field(default_factory= lambda : 0, init = False)
    _after_chi: float = field(default_factory= lambda : 0, init = False)
    _accepted_chi_squared: float = field(default_factory= lambda : np.inf, init = False)

    def set_delta_chi(self, delta_chi: float, after_chi: float) -> None:
        self.delta_chi = delta_chi
        self._after_chi = after_chi
    
    def is_acceptable(self) -> bool:
        return self._acceptance_state in [AcceptanceState.UNTESTED, AcceptanceState.ACCEPTABLE]
        
    def _calculate_success(self) -> None:
        not_unacceptable = self.is_acceptable()
        metropolis_success = lambda : False if self.temperature == 0 else self.rng_val < np.exp(-self.delta_chi / self.temperature)
        simulation_acceptable = self.delta_chi < 0 or metropolis_success() if not_unacceptable else False
        self._acceptance_state = AcceptanceState.ACCEPTABLE if simulation_acceptable else AcceptanceState.UNACCEPTABLE

    def handle_simulation(self, simulation: ScatteringSimulation) -> None:
        currently_acceptable = self.is_acceptable()
        old_chi_squared = simulation.current_goodness_of_fit
        new_chi_squared = simulation.get_goodness_of_fit() if currently_acceptable else old_chi_squared
        delta_chi = new_chi_squared - old_chi_squared
        self.set_delta_chi(delta_chi=delta_chi, after_chi = new_chi_squared)
        self._calculate_success()
        
        if self.is_acceptable():
            simulation.update_goodness_of_fit(new_chi_squared)
        self._accepted_chi_squared = simulation.current_goodness_of_fit
        
    def get_loggable_data(self) -> dict:
        return {
            "Current Chi^2": self._accepted_chi_squared,
            "Test Chi^2": self._after_chi,
            "RNG": self.rng_val,
            "Temperature": self.temperature,
            "Acceptable Move": self.is_acceptable(),
            "Acceptance Test": type(self).__name__,
        }