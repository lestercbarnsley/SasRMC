#%%

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc import constants
from sas_rmc.acceptance_scheme import MetropolisAcceptance, AcceptanceScheme
from sas_rmc.factories import parse_data

rng = constants.RNG

def create_metropolis_acceptance(temperature: float, cycle: int, step: int) -> MetropolisAcceptance:
    return MetropolisAcceptance(
        temperature=temperature,
        rng_val=rng.uniform(),
        loggable_data={
            "Cycle" : cycle,
            "Step" : step
        }
    )

@pydantic_dataclass
class AcceptanceFactory:
    annealing_stop_cycle_number: int
    anneal_start_temp: float
    annealing_type: str
    total_cycles: int
    anneal_fall_rate: float

    def create_very_fast_temperature(self, cycle: int) -> float:
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number > 0 else int(self.total_cycles / 2)
        if cycle > annealing_stop_cycle:
            return 0
        return self.anneal_start_temp * ((1- self.anneal_fall_rate)**cycle)
    
    def create_fast_temperature(self, cycle: int) -> float:
        annealing_stop_cycle = self.annealing_stop_cycle_number if self.annealing_stop_cycle_number > 0 else int(self.total_cycles / 2)
        if cycle > annealing_stop_cycle:
            return 0
        return self.anneal_start_temp / (1 + cycle)
    
    def get_temperature(self, cycle: int) -> float:
        if self.annealing_type.lower() == "greedy".lower():
            return 0
        if "very".lower() not in self.annealing_type.lower():
            return self.create_very_fast_temperature(cycle)
        return self.create_fast_temperature(cycle)
    
    def create_metropolis_acceptance(self, cycle: int, step: int) -> MetropolisAcceptance:
        temperature = self.get_temperature(cycle)
        return MetropolisAcceptance(
            temperature=temperature,
            rng_val=rng.uniform(),
            loggable_data={
                "Cycle" : cycle,
                "Step" : step
            }
        )

    def create_acceptance(self, cycle: int, step: int) -> AcceptanceScheme:
        return self.create_metropolis_acceptance(cycle, step)

    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return cls(**value_frame)


if __name__ == "__main__":
    print(create_metropolis_acceptance(0,0,0).rng_val)      

# %%
