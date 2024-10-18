#%%

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt, patches, figure, colors as mcolors
import numpy as np
import pandas as pd

from sas_rmc.vector import Vector
from sas_rmc.constants import PI


@dataclass
class LogCallback:

    @abstractmethod
    def start(self, document: dict  | None = None) -> None:
        pass

    @abstractmethod
    def event(self, document: dict  | None = None) -> None:
        pass

    @abstractmethod
    def stop(self, document: dict  | None = None) -> None:
        pass


@dataclass
class NoLogCallback(LogCallback):
    def start(self, document: dict | None = None) -> None:
        pass

    def event(self, document: dict | None = None) -> None:
        pass

    def stop(self, document: dict | None = None) -> None:
        pass

@dataclass
class PrintLogCallback(LogCallback):

    def start(self, document: dict | None = None) -> None:
        print('start', document)

    def event(self, document: dict | None = None) -> None:
        print('event', document)
        
    def stop(self, document: dict | None = None) -> None:
        print('event', document)


@dataclass
class QuietLogCallback(LogCallback):
    
    def start(self, document: dict | None = None) -> None:
        pass

    def event(self, document: dict | None = None) -> None:
        if document is None:
            return None
        doc = {k : f"{v:.6f}" if isinstance(v, float) else v for k, v in document.items() if k in ['Current goodness of fit', 'Cycle', 'Step']}
        if 'timestamp' in document:
            doc = doc | {'timestamp' : str(datetime.fromtimestamp(document.get('timestamp', 0)))}
        print(doc)
        

    def stop(self, document: dict | None = None) -> None:
        pass


@dataclass
class LogEventBus(LogCallback):
    log_callbacks: list[LogCallback]

    def start(self, document: dict  | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.start(document)

    def event(self, document: dict | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.event(document)

    def stop(self, document: dict | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.stop(document)


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
    particle_list: list
    dim_list: list

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
        box_volume = np.prod(self.dim_list)
        return float(total_particle_volume / box_volume)

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
        return float(
            self.get_scale_factor_value() * np.average(
                [box_data.calculate_concentration() for box_data in self.box_data_list]
                )
            )
    
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
        #start_sim_data = SimData.create_from_dict(self.start_docs[-1])

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
        detector_image_data_dfs = start_stop_doc_to_detector_image_data(document)
        for i, detector_data in enumerate(detector_image_data_dfs):
            fig = plot_detector_image(detector_data, fontsize=self.fontsize)
            fig.savefig(self.result_folder / Path(f"{self.file_plot_prefix}_detector_{i}_final.{self.file_plot_format}"))


def interpolate_qxqy(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, shadow_factor: np.ndarray, qx_target: float, qy_target: float) -> float | None:
    distances = np.sqrt((qx - qx_target)**2 + (qy - qy_target)**2)
    if not (qx.min() <= qx_target <= qx.max()):
        return None
    if not (qy.min() <= qy_target <= qy.max()):
        return None
    return intensity[distances.argmin()] if shadow_factor[distances.argmin()] else None

def sector_at_q(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, shadow_factor: np.ndarray, q_value: float, nominal_angle: float, angle_range: float) -> float | None:
    angles = np.linspace(-PI , +PI, num = 180)
    intensities = [interpolate_qxqy(qx, qy, intensity, shadow_factor, q_value * np.cos(angle), q_value * np.sin(angle)) for angle in angles if np.abs(angle - nominal_angle) < angle_range / 2 or np.abs(angle - (nominal_angle + PI)) < angle_range / 2]
    intensities = [intensity for intensity in intensities if intensity]
    if not intensities:
        return None
    return float(np.average(intensities))

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
        q, exp_int = sector_analysis(detector_df['qX'], detector_df['qY'], detector_df['intensity'], detector_df['shadow_factor'], sector_angle, PI / 10)
        q_sim, sim_int = sector_analysis(detector_df['qX'], detector_df['qY'], detector_df['simulated_intensity'], detector_df['shadow_factor'], sector_angle, PI / 10)
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
        detector_image_data_dfs = start_stop_doc_to_detector_image_data(document)
        for i, detector_data in enumerate(detector_image_data_dfs):
            fig = plot_profile(detector_data, sector_number=self.sector_number, fontsize=self.fontsize)
            fig.savefig(self.result_folder / Path(f"{self.file_plot_prefix}_profiles_{i}_final.{self.file_plot_format}"))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    fig.set_size_inches(4,4)
    d_0, d_1 = 14000, 14000
    ax.set_xlim(-d_0 / 2, +d_0 / 2)
    ax.set_ylim(-d_1 / 2, +d_1 / 2)

    ax.set_aspect("equal")

    ax.set_xlabel(r'X (Angstrom)',fontsize =  14)
    ax.set_ylabel(r'Y (Angstrom)',fontsize =  14)

    patch_list = [
            patches.Circle(
                xy = (0, 0),
                radius=120,
                ec = None,
                fc = 'black'
            ),
            patches.Circle(
                xy = (0, 0),
                radius=100,
                ec = None,
                fc = 'blue'
            )
        ]

    patch_list_2 = [
            patches.Circle(
                xy = (0,120 + 120),
                radius=120,
                ec = None,
                fc = 'black'
            ),
            patches.Circle(
                xy = (0,120 + 120),
                radius=100,
                ec = None,
                fc = 'blue'
            )
        ]

    for patch in patch_list + patch_list_2:
        #patch.set_snap(False)
        ax.add_patch(patch)



    #ax.set_box_aspect(d_1 / d_0)

    #fig.tight_layout()
    #print(fig.)
    fig.show()
    fig.savefig('test.pdf')

#%%