#%%

from dataclasses import dataclass
from pathlib import Path

from sas_rmc.loggers import LogCallback
from sas_rmc.loggers.excel_logger import SimData


@dataclass
class BoxPlotter(LogCallback):
    result_folder: Path
    file_plot_prefix: str
    file_plot_format: str = "pdf"
    fontsize: int = 14

    def start(self, document: dict | None = None) -> None:
        return super().start(document)

    def event(self, document: dict | None = None) -> None:
        return super().event(document)

    def stop(self, document: dict | None = None) -> None:
        stop_sim_data = SimData.create_from_dict(document)
        for i, box_data in enumerate(stop_sim_data.box_data_list):
            fig = box_data.to_plot(self.fontsize)
            fig.savefig(self.result_folder / Path(f"{self.file_plot_prefix}_particle_positions_box_{i}.{self.file_plot_format}"))

