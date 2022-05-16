#%%

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def generate_core_shell_template() -> Tuple[List[pd.DataFrame], List[str]]:
    data_frame_1 = pd.DataFrame(
        columns=["Parameter Name","Parameter value",None,None],
        data = [
            [None,None,None,None],
            ["simulation_title","this_is_a_test",r"# Name of output file",None],
            [None,None,r'# Cells containing "#" will be ignored by simulation',None],
            ["particle_type","CoreShellParticle",None,None],
            ["core_radius",None,r"# Angstroms",None],
            ["core_polydispersity",None,r"# Fraction",None],
            ["core_sld",None,r"# 10E-6 / Angstrom^2",None],
            ["shell_thickness",None,r"# Angstroms",None],
            ["shell_polydispersity",None,r"# Fraction",None],
            ["shell_sld",None,r"# 10E-6 / Angstrom^2",None],
            ["solvent_sld",None,r"# 10E-6 / Angstrom^2",None],
            ["core_magnetization",None,r"# Amperes/metre",None],
            [None,None,None,None],
            ["nominal_concentration",None,r"# vol/vol",r"# Put in information for two out of three of nominal_concentration, particle_number and box_number."],
            ["particle_number",None,r"# integer",r"# If all three are present, nominal_concentration will be ignored"],
            ["box_number",None,r"# integer",None],
            [None,None,None,None],
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
            [r"# Information about source for experimental, reduced data, either directly below or in the next tab",None,None,None],
            [r"# If you are fitting one detector image, you may input information here",None,None,None],
            [r"# If you are simultaneously fitting multiple detector images, use next tab, and make one row for each detector image (please don't delete headers)",None,None,None],
            [r"# Resolution parameter: Specify this parameter if detector smearing is ON and instrument resolution isn't included with reduced data",None,None,None],
            [None,None,None,None],
            [r"Data Source",None,r"# Either a file path pointing to an ASCII file containing 4 (or more) columns of reduced data or a name of a TAB in this excel file containing reduced data",None],
            [r"Label",None,r"# Optional, Recommended: A label for detector data used in the output report",None],
            [r"Polarization",None,r'# Optional: The polarization state used during this measurement, leave blank for "Unpolarized"',None],
            [r"Wavelength",None,r"# Optional: Angstroms, Resolution parameter",None],
            [r"Wavelength Spread",None,r"# Optional: Fraction, Resolution parameter",None],
            [r"Detector distance",None,r"# Optional: Metres, Resolution parameter",None],
            [r"Detector pixel",None,r"# Optional: Metres, Resolution parameter, if unknown, approximation acceptable",None],
            [r"Sample aperture",None,r"# Optional: Metre^2, Resolution parameter, if unknown, approximation acceptable",None],
            [r"Collimation distance",None,r"# Optional: Metres, Resolution parameter",None],
            [r"Collimation aperture",None,r"# Optional: Metre^2, Resolution parameter, if unknown, approximation acceptable",None],
            [r"Buffer Source",None,r"# Optional: A file path or a TAB label for buffer data. If buffer was already subtracted during data reduction, leave blank or specify ZERO. Enter numerical value to subtract constant buffer intensity from all pixels",None],
            [None,None,None,None],
            [r"# If experimental, reduced data is saved in subsequent TABs in this spreadsheet, the top row must contain headers",None,None,None],
            [r"# The following three headers are REQUIRED: qX, qY, intensity",None,None,None],
            [r"# The following headers are SUGGESTED: intensity_error, sigma_perp, sigma_para, shadow",None,None,None],
            [r"# The following headers may be INCLUDED, but serve no purpose: qZ",None,None,None],
            [None,None,None,None],
            [r"# Overwrite box dimensions. Do not do this unless you know what you're doing",None,None,None],
            [r"box_dimension_1",None,r"# Optional: Angstroms, strongly recommended to leave this blank",None],
            [r"box_dimension_2",None,r"# Optional: Angstroms, strongly recommended to leave this blank",None],
            [r"box_dimension_3",None,r"# Optional: Angstroms, strongly recommended to leave this blank",None]
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


if __name__ == "__main__":
    dfs, sheet_names = generate_core_shell_template()
    df_list_to_excel("_test.xlsx", dfs, sheet_names)

#%%
