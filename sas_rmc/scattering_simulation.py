#%%

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np



@dataclass
class SimulationParam:
    """Class for storing variables pertaining to specific simulation parameter

    Attributes
    ----------
    value : float
        The value of the parameter
    name : str
        A name identifier for the parameter which is typically unique within a set of SimulationParams
    bounds : Tuple[float, float]
        Upper and lower bounds for the parameter
    """
    value: float
    name: str
    bounds: Tuple[float, float] = (-np.inf, np.inf)

    def set_value(self, new_value: float) -> None:
        """Sets the value of a SimulationParam.

        Parameters
        ----------
        new_value : float
            The new value to be set
        """
        self.value = new_value

    def get_physical_acceptance(self) -> bool:
        """Determines if a SimulationParam is acceptable.

        The value is checked if it's between the bounds, returning True if it is.

        Returns
        -------
        bool
            True if the value is between the bounds
        """
        low_bound, high_bound = np.min(self.bounds), np.max(self.bounds)
        return low_bound <= self.value <= high_bound


@dataclass
class SimulationConstant(SimulationParam):
    def set_value(self, new_value: float) -> None:
        """Sets the value of a SimulationParam.

        This does nothing for a SimulationConstant, because the value for a constant shouldn't be changed.

        Parameters
        ----------
        new_value : float
            A test value that actually doesn't do anything
        """
        pass # Don't allow a constant to change

    def get_physical_acceptance(self) -> bool:
        return True


@dataclass
class SimulationParams:
    """Collection of SimulationParams.

    Exposes a set of methods to interact with the underlying list of SimulationParams.

    Attributes
    ----------
    params : List[SimulationParam]
        A list of SimulationParam.
    """
    
    params: List[SimulationParam]

    def __post__init__(self):
        name_list = []
        for i, param in enumerate(self.params):
            if param.name in name_list:
                param.name = param.name + f"_{i}"
            name_list.append(param.name) # No duplicate names in params

    @property
    def values(self) -> List[float]:
        """Get a list of values from SimulationParams.

        Returns
        -------
        List[float]
            The values contained inside each SimulationParam as a list.
        """
        return [param.value for param in self.params]

    def __getitem__(self, index: int) -> SimulationParam:
        return self.params[index]

    def get_physical_acceptance(self) -> bool:
        """Check that all values in the list of SimulationParam are within bounds

        Returns
        -------
        bool
            Returns True if all SimulationParam values are within bounds
        """
        return all(param.get_physical_acceptance() for param in self.params)

    def to_param_dict(self) -> dict:
        return {param.name : param for param in self.params}

    def to_value_dict(self) -> dict:
        return {param.name : param.value for param in self.params}

    def get_param(self, key: str) -> SimulationParam:
        return self.to_param_dict()[key] # Raises KeyError if key not there

    def get_value(self, key: str, default: Optional[float] = None) -> float:
        return self.to_value_dict().get(key, default)

    def set_value(self, key: str, value: float) -> None:
        param = self.get_param(key)
        param.set_value(value)


class Fitter(Protocol):
    def fit(self, simulation_params: SimulationParams) -> float:
        pass


@dataclass
class ScatteringSimulation:
    fitter: Fitter # The fitter is a strategy class
    simulation_params: SimulationParams# = field(default_factory=box_simulation_params_factory)
    current_goodness_of_fit: float = np.inf

    def get_goodness_of_fit(self) -> float:
        return self.fitter.fit(self.simulation_params)

    def update_goodness_of_fit(self, new_goodness_of_fit: float) -> None:
        self.current_goodness_of_fit = new_goodness_of_fit

    def get_physical_acceptance(self) -> bool: # Mark for deletion # Why is this marked for deletion? It's still being used.
        return self.simulation_params.get_physical_acceptance()


if __name__ == "__main__":
    pass

#%%