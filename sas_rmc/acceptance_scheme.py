
from enum import Enum
from dataclasses import asdict, dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from sas_rmc.constants import RNG

rng = RNG


class AcceptanceState(Enum):
    UNTESTED = "untested"
    ACCEPTABLE = "acceptable"
    UNACCEPTABLE = "unacceptable"


@dataclass
class AcceptanceScheme(ABC):
    @abstractmethod
    def is_acceptable(self, old_goodness_of_fit: float, new_goodness_of_fit: float) -> bool:
        pass

    @abstractmethod
    def get_loggable_data(self) -> dict:
        pass


@dataclass
class MetropolisAcceptance(AcceptanceScheme):
    temperature: float
    rng_val: float = field(default_factory= rng.uniform)
    loggable_data: dict | None = None
    
    def is_acceptable(self, old_goodness_of_fit: float, new_goodness_of_fit: float) -> bool:
        if np.abs(new_goodness_of_fit - 1) < np.abs(old_goodness_of_fit - 1):
            return True
        if self.temperature == 0:
            return False
        delta_chi = np.abs(new_goodness_of_fit - 1) - np.abs(old_goodness_of_fit - 1)
        return self.rng_val < np.exp(-delta_chi / self.temperature)
    
    def get_loggable_data(self) -> dict:
        loggable_data = self.loggable_data if self.loggable_data is not None else {}
        return {"Acceptance Scheme": type(self).__name__} |\
            loggable_data |\
            {k : v for k, v in asdict(self).items() if k != 'loggable_data'}
    

if __name__ == "__main__":
    pass