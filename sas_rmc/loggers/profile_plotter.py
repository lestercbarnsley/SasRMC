#%%
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors, figure

from sas_rmc import constants
from sas_rmc.loggers import LogCallback, excel_logger


PI = constants.PI


def interpolate_qxqy(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, shadow_factor: np.ndarray, qx_target: float, qy_target: float) -> float | None:
    if not (qx.min() <= qx_target <= qx.max()):
        return None
    if not (qy.min() <= qy_target <= qy.max()):
        return None
    distances = np.sqrt((qx - qx_target)**2 + (qy - qy_target)**2)
    if not shadow_factor[distances.argmin()]:
        return None
    return intensity[distances.argmin()]

def sector_at_q(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, shadow_factor: np.ndarray, q_value: float, nominal_angle: float, angle_range: float) -> float | None:
    angles = np.linspace(-PI , +PI, num = 180)
    intensities = [interpolate_qxqy(qx, qy, intensity, shadow_factor, q_value * np.cos(angle), q_value * np.sin(angle)) for angle in angles if np.abs(angle - nominal_angle) < angle_range / 2 or np.abs(angle - (nominal_angle + PI)) < angle_range / 2]
    intensities = [intensity for intensity in intensities if intensity]
    if not intensities:
        return None
    return np.average(intensities).item()

def sector_analysis(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, shadow_factor: np.ndarray, nominal_angle: float, angle_range: float) -> tuple[np.ndarray, np.ndarray]:
    qx_diff = np.diff(np.unique(qx)).max()
    qy_diff = np.diff(np.unique(qy)).max()
    q_step = np.min([qx_diff, qy_diff])
    q_max = np.sqrt(qx**2 + qy**2).max()
    q_range = np.arange(start = 0, stop=q_max, step = q_step)
    sector_intensity = [sector_at_q(qx, qy, intensity, shadow_factor, q, nominal_angle, angle_range) for q in q_range]
    q_final = [q for q, intensity in zip(q_range, sector_intensity) if intensity]
    intensity_final = [intensity for _, intensity in zip(q_range, sector_intensity) if intensity]
    return np.array(q_final), np.array(intensity_final)

def plot_profile(detector_df: pd.DataFrame, size_inches: tuple[float, float] = (6,6), sector_number: int = 4, fontsize: int = 16) -> figure.Figure:
    fig, ax = plt.subplots()
    w, h = size_inches
    fig.set_size_inches(w, h)
    sector_angles = np.linspace(start = 0, stop = PI / 2, num = sector_number)
    colours = [c for c in mcolors.BASE_COLORS.keys() if c!='w'] * 50
    factors = [i * 2 for i, _ in enumerate(sector_angles)]
    for factor, colour, sector_angle in zip(factors, colours, sector_angles):
        q, exp_int = sector_analysis(np.array(detector_df['qX']), np.array(detector_df['qY']), np.array(detector_df['intensity']), np.array(detector_df['shadow_factor']), sector_angle, PI / 10)
        q_sim, sim_int = sector_analysis(np.array(detector_df['qX']), np.array(detector_df['qY']), np.array(detector_df['simulated_intensity']), np.array(detector_df['shadow_factor']), sector_angle, PI / 10)
        ax.loglog(q, (10**factor) * exp_int, colour + '.' , label = f"{(180/PI)*sector_angle:.2f} deg")
        ax.loglog(q_sim, (10**factor) * sim_int, colour + '-')
    ax.set_xlabel(r'Q ($\AA^{-1}$)',fontsize =  fontsize)
    ax.set_ylabel(r'Intensity (cm $^{-1}$)',fontsize =  fontsize)
    ax.legend()
    return fig


@dataclass
class ProfilePlotter(LogCallback):
    result_folder: Path
    file_plot_prefix: str
    sector_number: int = 4
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
            fig = plot_profile(detector_data, sector_number=self.sector_number, fontsize=self.fontsize)
            fig.savefig(self.result_folder / Path(f"{self.file_plot_prefix}_profiles_{i}_final.{self.file_plot_format}"))
