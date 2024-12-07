
from typing import Sequence

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc.factories import parse_data
from sas_rmc.factories.box_factory import BoxFactory
from sas_rmc.scattering_simulation import ScatteringSimulation, SimulationParam


@pydantic_dataclass
class SimulationStateFactory:
    particle_factory: ParticleFactory
    box_factory: BoxFactory
    box_dimension_1: float
    box_dimension_2: float
    box_dimension_3: float

    def create_simulation_state(self, default_box_dimensions: Sequence[float] | None = None) -> ScatteringSimulation:
        box_dimensions = [self.box_dimension_1, self.box_dimension_2, self.box_dimension_3]
        if np.prod(box_dimensions) == 0:
            box_dimensions = default_box_dimensions
        if box_dimensions is None:
            raise ValueError("Box dimensions are missing.")
        return ScatteringSimulation(
            scale_factor=SimulationParam( 1.0, name = "scale_factor", bounds = (0, np.inf)),
            box_list=self.box_factory.create_box_list(self.create_particle, box_dimensions)
        )

    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame]):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        box_factory = BoxFactory.create_from_dataframes(dataframes)
        return cls(
            box_factory = box_factory,
            **value_frame
            )