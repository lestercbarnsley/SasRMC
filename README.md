# SasRMC
SasRMC is a Python library for numerical modelling of small-angle scattering data. 

## Requirements
**SasRMC will use Python 3.11 after the next push to main. Upgrade your Python installation today!**
SasRMC is compatible with Python 3.10 and newer. An upgrade to Python 3.11 is planned for very soon. Support for Python3.11 is planned until the end of 2025.
SaSRMC is dropping support for **conda**. Dependency management using **poetry** will be the preferred way going forward. Find out more here: https://python-poetry.org/

## Installation as a command-line tool

SasRMC can be used as a command-line tool or installed as a project dependency. Installation as a command-line tool is recommended using **pipx**.

1. The **pipx** package can either be installed using a package manager (see: https://pipx.pypa.io/stable/installation/), or downloaded as a stand-alone executable (https://github.com/pypa/pipx/releases). Using the stand-alone executable may be easier for Windows users.
2. If **pipx** has been installed, run:  
    '$ pipx install git+https://github.com/lestercbarnsley/SasRMC.git --verbose'  
    '$ pipx ensurepath'  
3. If **pipx** has been downloaded, navigate to the folder where pipx.pyz has been saved and run:  
    '$ python3.x pipx.pyz install git+https://github.com/lestercbarnsley/SasRMC.git --verbose'  
    '$ python3.x pipx.pyz ensurepath'  
where 3.x is the Python version you wish to use. (Please use 3.10 or greater). Restarting your terminal may be required.  
4. Run '$ sasrmc --version' to validate the installation. Run '$ sasrmc --help' at any time for assistance.  
5. Run '$ sasrmc config show' to see the current configuration.  
6. Run '$ sasrmc config -o [Output Folder]' to set a default output folder were the results of simulations will be saved.  
7. Run '$ sasrmc config -i [Input Spreadsheet File]' to set a default input *.xlsx file for simulations to use.  

## Installation as a Python dependency

This is more for developers who want to integrate SasRMC into their own projects.

1. Make an isolated Python project in your usual way.
2. If you're using **pip**, run:
    '(venv)$ pip install git+https://github.com/lestercbarnsley/SasRMC.git'
3. If you're using **poetry**, run:
    '(venv)$ poetry add git+https://github.com/lestercbarnsley/SasRMC.git'

## Getting simulation files
 


## Usage
Most configuration for SasRMC is done using Excel spreadsheets. You can use SasRMC without needing to edit any Python code.

1. In your text editor of choice, open `data/config.yaml`
2. Next to `input_config_source` specify an Excel spreadsheet that will contain configuration data for your simulation. Your excel file should be in the same folder as the `config.yaml` file.
3. Next to `output_folder` specify the folder that you want the output and log files to be saved to.
4. The `data` folder contains templates for how a typical simulation should be configured.
5. A new template Excel spreadsheet can be generated from the terminal by running one of the following commands:
    `(myenv)$ python main.py generate core shell template`
    `(myenv)$ python main.py generate dumbbell template`
    `(myenv)$ python main.py generate reload template`
More templates will be available in future versions.
6. Edit and save all config files. Please notice that Excel spreadsheets have multiple tabs for additional options.
7. In the terminal, run
    `(myenv)$ python main.py`
8. When the simulation is complete, you can find all outputs in the specified `output_folder`
9. A simulation can be finished early at any time with the keyboard shortcut `Ctrl+C`

The `/data` folder contains a file `CoreShell_F20_pol.xlsx` which contains data described in the associated publication, and shows an example for how a simulation for a SANSPol measurement across 3 detector configurations and 2 polarization states can be set out.

## Terms of Use
SasRMC is released for free under the MIT license. An associated publication is available here:

Barnsley, L.C., Nandakumaran, N., Feoktystov, A., Dulle, M., Fruhner, L. & Feygenson, M. (2022). J. Appl. Cryst. 55,
https://doi.org/10.1107/S1600576722009219.

If you find that using SasRMC has added value to your scientific research, the authors would like to ask you to consider including a reference to the above work in your publication.

## Advanced Usage

1. See instructions about "Installation as a Python dependency" 
2. Type: `import sas_rmc`
3. More documentation coming soon...
