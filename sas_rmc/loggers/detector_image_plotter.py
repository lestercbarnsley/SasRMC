#%%
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, figure
import pandas as pd

from sas_rmc.loggers import LogCallback, excel_logger


def plot_detector_image(detector_df: pd.DataFrame, fontsize = 14, size_inches: tuple[float, float] = (6,6)) -> figure.Figure:
    fig, ax = plt.subplots()
    w, h = size_inches
    fig.set_size_inches(w, h)

    qx_lin = detector_df['qX']
    qy_lin = detector_df['qY']
    intensity_lin = detector_df['intensity']
    intensity_sim_lin = detector_df['simulated_intensity']
    qx_diff = np.diff(np.unique(qx_lin)).max()
    qy_diff = np.diff(np.unique(qy_lin)).max()
    qx, qy = np.meshgrid(
        np.arange(start=qx_lin.min(), stop = qx_lin.max(), step=qx_diff),
        np.arange(start=qy_lin.min(), stop = qy_lin.max(), step=qy_diff),
        )
    intensity = np.zeros(qx.shape)

    for qxi, qyi, intensity_i, intensity_sim_i in zip(qx_lin, qy_lin, intensity_lin, intensity_sim_lin):
        j = ((qy[:,0] - qyi)**2).argmin()
        i = ((qx[0,:] - qxi)**2).argmin()
        intensity[j, i] = intensity_i if qxi < 0 else intensity_sim_i
    
    
    ax.pcolormesh(qx, qy, intensity, cmap='jet', norm='log')
    ax.set_xlabel(r'Q$_{x}$ ($\AA^{-1}$)',fontsize =  fontsize)#'x-large')
    ax.set_ylabel(r'Q$_{y}$ ($\AA^{-1}$)',fontsize =  fontsize)#'x-large')
    
    ax.axhline(0, linestyle='-', color='k') # horizontal lines
    ax.axvline(0, linestyle='-', color='k') # vertical lines
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


@dataclass
class DetectorImagePlotter(LogCallback):
    result_folder: Path
    file_plot_prefix: str
    file_plot_format: str = "pdf"
    fontsize: int = 16

    def start(self, document: dict | None = None) -> None:
        return super().start(document)
    
    def event(self, document: dict | None = None) -> None:
        return super().event(document)
    
    def stop(self, document: dict | None = None) -> None:
        if document is None:
            return None
        detector_image_data_dfs = excel_logger.start_stop_doc_to_detector_image_data(document)
        for i, detector_data in enumerate(detector_image_data_dfs):
            fig = plot_detector_image(detector_data, fontsize=self.fontsize)
            fig.savefig(self.result_folder / Path(f"{self.file_plot_prefix}_detector_{i}_final.{self.file_plot_format}"))
