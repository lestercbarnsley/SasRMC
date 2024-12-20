#%%
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from sas_rmc import Particle
from sas_rmc.box_simulation import Box


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
    bounds: tuple[float, float] = (-np.inf, np.inf)

    def set_value(self, new_value: float) -> Self:
        return type(self)(value = new_value, name = self.name, bounds=self.bounds)

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
    
    def get_loggable_data(self) -> dict:
        return {
            "Param name" : self.name,
            "Value" : self.value,
            "Bound lower" : self.bounds[0],
            "Bound upper" : self.bounds[1]
        }


@dataclass
class SimulationConstant(SimulationParam):
    def set_value(self, new_value: float) -> SimulationParam:
        return self

    def get_physical_acceptance(self) -> bool:
        return True
    
    def get_loggable_data(self) -> dict:
        return super().get_loggable_data()


@dataclass
class ScatteringSimulation:
    scale_factor: SimulationParam
    box_list: list[Box]

    def get_physical_acceptance(self) -> bool:
        if not self.scale_factor.get_physical_acceptance():
            return False
        if any(box.collision_test() for box in self.box_list):
            return False
        return True
    
    def set_scale_factor(self, new_scale_factor: float) -> Self:
        return type(self)(
            scale_factor=self.scale_factor.set_value(new_scale_factor),
            box_list = self.box_list
        )
    
    def get_particle(self, box_index: int, particle_index: int) -> Particle:
        return self.box_list[box_index].particles[particle_index]
    
    def change_particle(self, box_index: int, particle_index: int, new_particle: Particle) -> Self:
        return type(self)(
            scale_factor=self.scale_factor,
            box_list=[
                box if box_index !=i else box.change_particle(particle_index, new_particle) 
                for i, box in enumerate(self.box_list)]
        )
    
    def get_loggable_data(self) -> dict:
        return {
            'scattering_simulation' : {
                f'box_{i}' : box.get_loggable_data() 
                for i, box 
                in enumerate(self.box_list)
                },
            'scale_factor' : self.scale_factor.get_loggable_data()
        }



if __name__ == "__main__":
    pass

#%%