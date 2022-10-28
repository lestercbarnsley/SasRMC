#%%

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class AnnealingConfig(ABC):

    @abstractmethod
    def get_temperature(self, cycle: int) -> float:
        pass


@dataclass
class GreedyAnneal(AnnealingConfig):

    def get_temperature(self, cycle: int) -> float:
        return 0.0


@dataclass
class FastAnneal(AnnealingConfig):
    annealing_stop_cycle_number: int
    anneal_start_temp: float

    def get_temperature(self, cycle: int) -> float:
        if cycle > self.annealing_stop_cycle_number:
            return 0
        return self.anneal_start_temp / (cycle + 1)


@dataclass
class VeryFastAnneal(AnnealingConfig):
    annealing_stop_cycle_number: int
    anneal_start_temp: float
    anneal_fall_rate: float

    def get_temperature(self, cycle: int) -> float:
        if cycle > self.annealing_stop_cycle_number:
            return 0
        return self.anneal_start_temp * np.exp(-cycle * self.anneal_fall_rate)


TYPE_DICT = {
    "Very Fast" : lambda annealing_stop_cycle_number, anneal_start_temp, anneal_fall_rate: VeryFastAnneal(annealing_stop_cycle_number, anneal_start_temp, anneal_fall_rate),
    "Very fast" : lambda annealing_stop_cycle_number, anneal_start_temp, anneal_fall_rate: VeryFastAnneal(annealing_stop_cycle_number, anneal_start_temp, anneal_fall_rate),
    "Fast" : lambda annealing_stop_cycle_number, anneal_start_temp, anneal_fall_rate: FastAnneal(annealing_stop_cycle_number, anneal_start_temp),
    "Greedy" : lambda annealing_stop_cycle_number, anneal_start_temp, anneal_fall_rate: GreedyAnneal(),
}

def gen_from_dict(d: dict) -> AnnealingConfig:
    annealing_type = d.get("annealing_type", "Very Fast")
    annealing_stop_cycle_number = d.get("annealing_stop_cycle_number", int(d.get("total_cycles") / 2))
    anneal_start_temp = d.get("anneal_start_temp", 10.0)
    annealing_fall_rate = d.get("anneal_fall_rate", 0.1)
    return TYPE_DICT[annealing_type](annealing_stop_cycle_number, anneal_start_temp, annealing_fall_rate)




