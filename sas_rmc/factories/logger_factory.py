

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from pydantic.dataclasses import dataclass as pydantic_dataclass

from sas_rmc import loggers
from sas_rmc.factories import parse_data
from sas_rmc.factories.evaluator_factory import ProfileType, infer_profile_type
from sas_rmc.loggers.logger import LogCallback


@pydantic_dataclass
class LoggerFactory(ABC):
    @abstractmethod
    def create_callbacks(self) -> LogCallback:
        pass


@pydantic_dataclass
class CallbackFactory(LoggerFactory):
    results_folder: Path
    simulation_title: str
    datetime_string: str
    output_plot_format: str = "PDF"
    sector_number: int = 4
    fontsize: int = 16

    def create_callbacks(self) -> LogCallback:
        results_folder = self.results_folder
        datetime_string = self.datetime_string
        return loggers.LogEventBus(
            log_callbacks=[
                loggers.CLIckLogger(), 
                loggers.ExcelCallback(excel_file= results_folder / Path(f'{datetime_string}_{self.simulation_title}.xlsx')),
                loggers.BoxPlotter(result_folder=results_folder, file_plot_prefix=f'{datetime_string}_{self.simulation_title}', file_plot_format=self.output_plot_format, fontsize=self.fontsize),
                loggers.DetectorImagePlotter(result_folder=results_folder, file_plot_prefix=f'{datetime_string}_{self.simulation_title}', file_plot_format=self.output_plot_format, fontsize=self.fontsize),
                loggers.ProfilePlotter(result_folder=results_folder, file_plot_prefix=f'{datetime_string}_{self.simulation_title}', file_plot_format=self.output_plot_format, sector_number= self.sector_number, fontsize=self.fontsize)
                ]
        )
    
    @classmethod
    def create_from_dataframes(cls, dataframes: dict[str, pd.DataFrame], results_folder: Path):
        value_frame = parse_data.parse_value_frame(dataframes['Simulation parameters'])
        return CallbackFactory(results_folder = results_folder, **value_frame)
    

def create_logger_from_dataframes(dataframes: dict[str, pd.DataFrame], results_folder: Path) -> LoggerFactory:
    profile_type = infer_profile_type(dataframes)
    if profile_type == ProfileType.DETECTOR_IMAGE:
        return CallbackFactory.create_from_dataframes(dataframes, results_folder)
    raise NotImplementedError("to do")

