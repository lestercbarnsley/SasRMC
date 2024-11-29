#%%
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import patches, pyplot as plt, figure
import pandas as pd

from sas_rmc import Vector
from sas_rmc.loggers import LogCallback


def particle_data_to_magnetization_vector(particle_data: dict) -> Vector | None:
    try:
        return Vector(
            x = particle_data['Magnetization.X'],
            y = particle_data['Magnetization.Y'],
            z = particle_data['Magnetization.Z']
        )
    except KeyError:
        return None
    
def particle_data_to_patches(particle_data: dict) -> list[patches.Patch]:
    particle_type = particle_data.get('Particle type')
    if particle_type == 'CoreShellParticle':
        xy = (particle_data.get('Position.X', -np.inf), particle_data.get('Position.Y', -np.inf))
        core_radius = particle_data.get('Core radius', 0.0)
        shell_radius = core_radius + particle_data.get('Thickness')
        return [
            patches.Circle(
                xy = xy,
                radius=shell_radius,
                ec = None,
                fc = 'black'
            ),
            patches.Circle(
                xy = xy,
                radius=core_radius,
                ec = None,
                fc = 'blue'
            )
        ]
    else:
        raise ValueError("Unrecognized particle type")
    
def magnetic_data_to_patches(particle_data: dict) -> list[patches.Patch]:
    particle_type = particle_data.get('Particle type')
    if particle_type == 'CoreShellParticle':
        x, y = particle_data.get('Position.X', -np.inf), particle_data.get('Position.Y', -np.inf)
        core_radius = particle_data.get('Core radius', 0.0)
        magnetization = particle_data_to_magnetization_vector(particle_data)
        if magnetization is None:
            return []
        dx, dy, *_ = (3 * core_radius * magnetization.unit_vector).to_tuple()
        return [
            patches.Arrow(x = x, y=y, dx=dx, dy=dy)
        ]
    else:
        raise ValueError("Unrecognized particle type")


@dataclass
class BoxData:
    particle_list: list[dict]
    dim_list: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        dim_data = {f'box_dimension_{i}' : dim for i, dim in enumerate(self.dim_list)}
        particle_data_list = [particle_data | (dim_data if i == 0 else {}) for i, particle_data in enumerate(self.particle_list)]
        return pd.DataFrame(data = particle_data_list)
    
    def get_all_magnetization(self) -> list[Vector]:
        all_vectors = [particle_data_to_magnetization_vector(particle_data) for particle_data in self.particle_list]
        return [vector for vector in all_vectors if vector is not None]
    
    def get_all_volume(self) -> list[float]:
        return [particle['Volume'] for particle in self.particle_list]
    
    def to_plot(self, fontsize: int = 14, size_inches: tuple[float, float] = (6,6), include_magnetic: bool = False) -> figure.Figure:
        fig, ax = plt.subplots()
        w, h = size_inches
        fig.set_size_inches(w, h)
                    
        d_0, d_1 = self.dim_list[0], self.dim_list[1]
        ax.set_xlim(-d_0 / 2, +d_0 / 2)
        ax.set_ylim(-d_1 / 2, +d_1 / 2)

        ax.set_aspect("equal")

        ax.set_xlabel(r'X (Angstrom)',fontsize =  fontsize)
        ax.set_ylabel(r'Y (Angstrom)',fontsize =  fontsize)

        for particle_data in self.particle_list:
            for patch in particle_data_to_patches(particle_data):
                ax.add_patch(patch)

        if include_magnetic:
            for particle_data in self.particle_list:
                for patch in magnetic_data_to_patches(particle_data):
                    ax.add_patch(patch)

        fig.tight_layout()
        return fig

    def calculate_concentration(self) -> float:
        total_particle_volume = np.sum(self.get_all_volume())
        box_volume = np.prod(self.dim_list).item()
        return total_particle_volume / box_volume

    @classmethod
    def create_from_dict(cls, d):
        particle_list = []
        dim_list = []
        for key in d:
            if 'particle' in key.lower():
                particle_list.append(d[key])
            elif 'dim' in key.lower():
                dim_list.append(d[key])
        return BoxData(
            particle_list=particle_list,
            dim_list=dim_list
        )
    

@dataclass
class SimData:
    box_data_list: list[BoxData]
    scale_factor_data: dict

    def to_dataframes(self) -> list[pd.DataFrame]:
        return [box_data.to_dataframe() for box_data in self.box_data_list]
    
    def get_scale_factor_value(self) -> float:
        return self.scale_factor_data['Value']

    def calculate_corrected_concentration(self) -> float:
        return self.get_scale_factor_value() * np.average(
                [box_data.calculate_concentration() for box_data in self.box_data_list]
                ).item()
    
    def get_average_magnetization(self) -> Vector:
        vectors = sum((box_data.get_all_magnetization() for box_data in self.box_data_list), start= [])
        total_particles = len(vectors)
        vector_sum = sum(vectors, start = Vector.null_vector())
        return vector_sum / total_particles

    @classmethod
    def create_from_dict(cls, d):
        simulation_data = d.get('scattering_simulation', {})
        box_data_list = []
        scale_factor_data = d.get('scale_factor', {})
        for key in simulation_data:
            if 'box_' in key.lower():
                box_data_list.append(BoxData.create_from_dict(simulation_data[key]))
        return SimData(box_data_list=box_data_list, scale_factor_data=scale_factor_data)


def event_docs_to_simulation_data(event_docs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(data = event_docs)

def fitter_data_to_detector_data(data: dict) -> list[dict]:
    detector_data = data['Detector data']
    simulated_intensity = data['Simulated intensity']
    polarization = data['Polarization']
    return [
        det_data | {'simulated_intensity' : sim_intensity} | ({'Polarization' : polarization} if i == 0 else {})
        for i, (det_data, sim_intensity) in enumerate(zip(detector_data, simulated_intensity))
        ]

def start_stop_doc_to_detector_image_data(doc: dict) -> list[pd.DataFrame]:
    fitter_doc = doc['Fitter']
    return [
        pd.DataFrame(fitter_data_to_detector_data(fitter_doc[fitter_key])) 
        for fitter_key in sorted(fitter_doc.keys()) 
        if 'fitter' in fitter_key.lower()
    ]

def sim_data_to_global_df(sim_data: SimData, total_time: float) ->  pd.DataFrame:
    
    final_rescale = sim_data.get_scale_factor_value()
    estimated_concentration = sim_data.calculate_corrected_concentration()

    average_magnetization = sim_data.get_average_magnetization()
    
    d = {
        "Final scale": [final_rescale],
        "Estimated concentration (v/v)": [estimated_concentration],
        "Simulation time (s)": [total_time],
        "AverageMagnetization.X" : [average_magnetization.x],
        "AverageMagnetization.Y" : [average_magnetization.y],
        "AverageMagnetization.Z" : [average_magnetization.z]
    }
    return pd.DataFrame(d)


@dataclass
class ExcelCallback(LogCallback):
    excel_file: Path
    start_docs: list[dict] = field(default_factory=list)
    event_docs: list[dict] = field(default_factory=list)
    stop_docs: list[dict] = field(default_factory=list)

    def start(self, document: dict | None = None) -> None:
        if document:
            self.start_docs.append(document)

    def event(self, document: dict | None = None) -> None:
        if document:
            self.event_docs.append(document)

    def stop(self, document: dict | None = None) -> None:
        if document:
            self.stop_docs.append(document)
        with pd.ExcelWriter(self.excel_file) as writer:
            start_sim_data = SimData.create_from_dict(self.start_docs[-1])
            for box_number, initial_particle_state_df in enumerate(start_sim_data.to_dataframes()):
                initial_particle_state_df.to_excel(writer, sheet_name=f"Box {box_number} Initial Particle States")
            simulation_df = event_docs_to_simulation_data(self.event_docs)
            simulation_df.to_excel(writer, sheet_name="Simulation Data")
            final_sim_data = SimData.create_from_dict(self.stop_docs[-1])
            for box_number, final_particle_state_df in enumerate(final_sim_data.to_dataframes()):
                final_particle_state_df.to_excel(writer, sheet_name=f"Box {box_number} Final Particle States")
            detector_image_data_dfs = start_stop_doc_to_detector_image_data(self.stop_docs[-1])
            for detector_number, detector_data_df in enumerate(detector_image_data_dfs):
                detector_data_df.to_excel(writer, sheet_name= f"Final detector image {detector_number}")
            total_time = max(event_doc.get('timestamp', 0) for event_doc in self.event_docs) - min(event_doc.get('timestamp', np.inf) for event_doc in self.event_docs)
            
            global_df = sim_data_to_global_df(final_sim_data, total_time = total_time)
            global_df.to_excel(writer, sheet_name="Global parameters Final")


if __name__ == "__main__":
    pass
