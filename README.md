# SasRMC
SasRMC is a Python library for numerical modelling of small-angle scattering data. 

## Setup
**SasRMC will use Python 3.11 after the next push to main. Upgrade your Python installation today!**
SasRMC is compatible with Python 3.10 and newer. An upgrade to Python 3.11 is planned for very soon.
SaSRMC is dropping support for **conda**. Dependency management using **poetry** will be the preferred way going forward. Find out more here: https://python-poetry.org/

1. Download source files from https://github.com/lestercbarnsley/SasRMC -> `Code` -> `Download ZIP`
2. Install **poetry**. Check the virtual environment and/or base Python installation version with 
    `$ poetry env info`
If the version is not compatible with the required Python version (currently 3.10 and newer) you will need to upgrade your virtual environment and/or base Python.
3. Navigate to the folder that contains `pyproject.toml`. Run
    `$ poetry install`
    

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

1. Open a new .py file
2. Type: `import sas_rmc`
3. More documentation coming soon...
