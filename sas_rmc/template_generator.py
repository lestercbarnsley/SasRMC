#%%

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sas_rmc.particles import CoreShellParticle, DumbbellParticle

@dataclass
class DataRow:
    key: str = ""
    value: str | int | float = ""
    unit_hint: str = ""
    other_comments: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "Parameter Name" : self.key,
            "Parameter value" : self.value,
            "Unit hint" : r"# " + self.unit_hint} | {
                f"Hint {i+1}": r"# " + comment for 
                i, comment in enumerate(self.other_comments)}
    

def data_rows_to_df(data_rows: list[DataRow]) -> pd.DataFrame:
    return pd.DataFrame(data = [data_row.to_row() for data_row in data_rows])



ALWAYS_PRESENT_DATA = [
    DataRow(),
    DataRow("simulation_title", "write_title_here",unit_hint="", other_comments=["Name of output file"] ),
    DataRow("", "", "", other_comments=['Cells containing "#" will be ignored by simulation']),
    DataRow()
]

ANGSTROM_UNIT = "Angstroms"
FRACTION_UNIT = "Fraction"
SLD_UNIT = "10E-6 / Angstrom^2"
FIELD_UNIT = "Amperes/metre"

CORE_SHELL_PARTICLE_DATA = [
    DataRow("particle_type", CoreShellParticle.__name__, other_comments=["The form factor for this particle type is calculated analytically"]),
    DataRow("core_radius", unit_hint=ANGSTROM_UNIT),
    DataRow("core_polydispersity", unit_hint=FRACTION_UNIT),
    DataRow("core_sld", unit_hint=SLD_UNIT),
    DataRow("shell_thickness", unit_hint=ANGSTROM_UNIT),
    DataRow("shell_polydispersity", unit_hint=FRACTION_UNIT),
    DataRow("shell_sld", unit_hint=SLD_UNIT),
    DataRow("solvent_sld", unit_hint=SLD_UNIT),
    DataRow("core_magnetization", value=0.0, unit_hint=FIELD_UNIT, other_comments=["Must be zero for SAXS"]),
    DataRow(),
]

DUMBBELL_PARTICLE_DATA = [
    DataRow("particle_type", DumbbellParticle.__name__, other_comments=["The form factor for this particle type is calculated numerically"]),
    DataRow("core_radius", unit_hint=ANGSTROM_UNIT),
    DataRow("core_polydispersity", unit_hint=FRACTION_UNIT),
    DataRow("core_sld", unit_hint=SLD_UNIT),
    DataRow("seed_radius", unit_hint=ANGSTROM_UNIT),
    DataRow("seed_polydispersity", unit_hint=FRACTION_UNIT),
    DataRow("seed_sld", unit_hint=SLD_UNIT),
    DataRow("shell_thickness", unit_hint=ANGSTROM_UNIT),
    DataRow("shell_polydispersity", unit_hint=FRACTION_UNIT),
    DataRow("shell_sld", unit_hint=SLD_UNIT),
    DataRow("solvent_sld", unit_hint=SLD_UNIT),
    DataRow("core_magnetization", value=0.0, unit_hint=FIELD_UNIT, other_comments=["Must be zero for SAXS"]),
    DataRow("seed_magnetization", value=0.0, unit_hint=FIELD_UNIT, other_comments=["Must be zero for SAXS"]),
    DataRow()
    ]
RELOAD_PARTICLE_DATA = [
    DataRow("particle_type", DumbbellParticle.__name__),
    DataRow("log_file_source", unit_hint="File path", other_comments=["Point to a previously saved log file (.xlsx). Particle configurations and detector data will be loaded directly from the log file. Simulation configurations are loaded from below parameters"]),
    DataRow()
    ]

VOLVOL = "vol/vol"
INTEGER = "integer"

BOX_DATA = [
    DataRow("nominal_concentration", unit_hint=VOLVOL, other_comments=[r"# Put in information for two out of three of nominal_concentration, particle_number and box_number."]),
    DataRow("particle_number", unit_hint=INTEGER, other_comments=[r"# If all three are present, nominal_concentration will be ignored"]),
    DataRow("box_number", unit_hint=INTEGER),
    DataRow()
    ]

SIMULATION_DATA = [
    DataRow("total_cycles", unit_hint=INTEGER),
    DataRow("annealing_type", "Very fast","Acceptable options: Fast, Very Fast, Greedy" ),
    DataRow("anneal_start_temp", 10, other_comments=["If uncertain, leave as is"]),
    DataRow("anneal_fall_rate", 0.1, other_comments=["If uncertain, leave as is"]),
    DataRow("annealing_stop_cycle_number", other_comments=["Leave this blank to default to 50% of the total cycles"]),
    DataRow()]

DETECTOR_SETTINGS = [
    DataRow("detector_smearing", "ON", unit_hint="Acceptable options: ON, OFF"),
    DataRow("field_direction", "Y", unit_hint="Acceptable options: X, Y, Z, OFF"),
    DataRow()
]

RESULT_OPTIONS = [
    DataRow("force_log_file", "ON", unit_hint="Acceptable options: ON, OFF, Forces log file to be saved, even if simulation ended prematurely"),
    DataRow("output_plot_format", "PDF", unit_hint="Acceptable options: NONE, PDF, PNG, JPG, Saves collection of images to review after simulation. These are NOT publication quality figures."),
    DataRow()
]

'''            
    ["total_cycles",None,r"# integer",None],
    ["annealing_type",r"Very fast",r"# Acceptable options: Fast, Very Fast, Greedy",None],
    ["anneal_start_temp",10,r"# If uncertain, leave as is",None],
    ["anneal_fall_rate",0.1,r"# If uncertain, leave as is",None],
    ["annealing_stop_cycle_number",None,r"# Leave this blank to default to 50% of the total cycles",None],
    [None,None,None,None],
    ["detector_smearing",r"ON",r"# Acceptable options: ON, OFF",None],
    ["field_direction",r"Y",r"# Acceptable options: X, Y, Z, OFF",None],
    ["force_log_file",r"ON",r"# Acceptable options: ON, OFF, Forces log file to be saved, even if simulation ended prematurely",None],
    ["output_plot_format",r"PDF",r"# Acceptable options: NONE, PDF, PNG, JPG, Saves collection of images to review after simulation. These are NOT publication quality figures.",None],
    [None,None,None,None],
'''

DETECTOR_DATA = [
    DataRow(other_comments=["Information about source for experimental, reduced data, either directly below or in the next tab"]),
    DataRow(other_comments=["If you are fitting one detector image, you may input information here"]),
    DataRow(other_comments=["If you are simultaneously fitting multiple detector images, use next tab, and make one row for each detector image (please don't delete headers)"]),
    DataRow(other_comments=["Resolution parameter: Specify this parameter if detector smearing is ON and instrument resolution isn't included with reduced data"]),
    DataRow(),
    DataRow("Data Source", unit_hint="Either a file path pointing to an ASCII file containing 4 (or more) columns of reduced data or a name of a TAB in this excel file containing reduced data"),
    DataRow("Label", unit_hint="Optional, Recommended: A label for detector data used in the output report"),
    DataRow("Polarization", unit_hint='Optional: The polarization state used during this measurement, leave blank for "Unpolarized"'),
    DataRow("Wavelength", unit_hint="Optional: Angstroms, Resolution parameter"),
    DataRow("Wavelength Spread", unit_hint="Optional: Fraction, Resolution parameter"),
    DataRow("Detector distance", unit_hint="Optional: Metres, Resolution parameter"),
    DataRow("Detector pixel", unit_hint="Optional: Metres, Resolution parameter, if unknown, approximation acceptable"),
    DataRow("Sample aperture", unit_hint="Optional: Metre^2, Resolution parameter, if unknown, approximation acceptable"),
    DataRow("Collimation distance", unit_hint="Optional: Metres, Resolution parameter"),
    DataRow("Collimation aperture", unit_hint="Optional: Metre^2, Resolution parameter, if unknown, approximation acceptable"),
    DataRow("Buffer Source", unit_hint="Optional: A file path or a TAB label for buffer data. If buffer was already subtracted during data reduction, leave blank or specify ZERO. Enter numerical value to subtract constant buffer intensity from all pixels"),
    DataRow(),
    DataRow(other_comments=["If experimental, reduced data is saved in subsequent TABs in this spreadsheet, the top row must contain headers"]),
    DataRow(other_comments=["The following three headers are REQUIRED: qX, qY, intensity"]),
    DataRow(other_comments=["The following headers are SUGGESTED: intensity_error, sigma_perp, sigma_para, shadow"]),
    DataRow(other_comments=["The following headers may be INCLUDED, but serve no purpose: qZ"]),
    DataRow(),
    DataRow("box_dimension_1", unit_hint="Optional: Angstroms, strongly recommended to leave this blank"),
    DataRow("box_dimension_2", unit_hint="Optional: Angstroms, strongly recommended to leave this blank"),
    DataRow("box_dimension_3", unit_hint="Optional: Angstroms, strongly recommended to leave this blank"),
    DataRow()
]

def generate_normal_template(particle_data: List[List[str]], simulation_data: List[List[str]] = SIMULATION_DATA) -> Tuple[List[pd.DataFrame], List[str]]:
    data_frame_1 = pd.DataFrame(
        columns=["Parameter Name","Parameter value",None,None],
        data = [
            *ALWAYS_PRESENT_DATA,
            *particle_data,
            [None, None, None, None],
            *simulation_data
        ]
    )
    columns = [
        "Data Source",
        "Label",
        "Polarization",
        "Wavelength",
        "Wavelength Spread",
        "Detector distance",
        "Detector pixel"
        ,"Sample aperture",
        "Collimation distance",
        "Collimation aperture",
        "Buffer Source"]
    
    data_frame_2 = pd.DataFrame(
        columns=columns,
        data = [[None for _, _ in enumerate(columns)]]
    )

    data_cols = [
        "qX", 
        "qY",
        "intensity",
        "intensity_error"]
    data_frame_3 = pd.DataFrame(
        columns=data_cols,
        data = [[None for _,_ in enumerate(data_cols)]]
    )
    sheet_names = [
        "Simulation parameters",
        "Data parameters",
        "Experimental data 1"
        ]
    return [data_frame_1, data_frame_2, data_frame_3], sheet_names

def df_list_to_excel(out_file: Path, dataframe_list: List[pd.DataFrame], sheet_names: List[str] = None) -> None:
    s_names = sheet_names if sheet_names is not None else [None for _ in dataframe_list]
    with pd.ExcelWriter(out_file) as writer:
        for df, sheet_name in zip(dataframe_list, s_names):
            df.to_excel(writer, sheet_name=sheet_name, index = False)

def generate_core_shell(output_path: Path) -> None:
    particle_data = CORE_SHELL_PARTICLE_DATA
    simulation_data = [*BOX_DATA,*SIMULATION_DATA, *DETECTOR_DATA]
    dfs, sheet_names = generate_normal_template(particle_data=particle_data, simulation_data=simulation_data)
    df_list_to_excel(output_path, dfs, sheet_names)

def generate_dumbbell(output_path: Path) -> None:
    particle_data = DUMBBELL_PARTICLE_DATA
    simulation_data = [*BOX_DATA,*SIMULATION_DATA, *DETECTOR_DATA]
    dfs, sheet_names = generate_normal_template(particle_data=particle_data, simulation_data=simulation_data)
    df_list_to_excel(output_path, dfs, sheet_names)

def generate_reload(output_path: Path) -> None:
    particle_data = RELOAD_PARTICLE_DATA
    simulation_data = SIMULATION_DATA
    dfs, sheet_names = generate_normal_template(particle_data=particle_data, simulation_data=simulation_data)
    df_list_to_excel(output_path, dfs, sheet_names)

def generate_template(command: str, output_path: Path) -> None:
    TEMPLATE_PARAMS = {
    "generate core shell template":{
        "template_name" : "CoreShell_Simulation_Input",
        "template_generating_method" : generate_core_shell
        },
    "generate dumbbell template":{
        "template_name" : "Dumbell_Simulation_Input",
        "template_generating_method" : generate_dumbbell
        },
    "generate reload template":{
        "template_name" : "Reload_Simulation_Input",
        "template_generating_method" : generate_reload
        }
    }
    template_generating_method = TEMPLATE_PARAMS[command]["template_generating_method"]
    template_generating_method(output_path)

if __name__ == "__main__":
    dfs, sheet_names = generate_normal_template(particle_data=CORE_SHELL_PARTICLE_DATA)
    df_list_to_excel("_test.xlsx", dfs, sheet_names)

#%%
