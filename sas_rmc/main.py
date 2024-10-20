#%%
from pathlib import Path

import click
import yaml

from sas_rmc.factories import runner_factory

CONFIG_FILE = Path(__file__).parent.parent / Path("data") / Path("config.yaml")

with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DEFAULT_INPUT = Path(config.get('input_config_source'))
DEFAULT_OUTPUT =  Path(config.get('output_folder'))


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx: click.Context):
    '''SasRMC is a Python library for numerical modelling of small-angle scattering data.'''
    pass

@click.command()
@click.option("-i", "--input", "inputs", help="Excel file(s) containing small-angle scattering data and simulation configurations to run", type = click.Path(), default = [DEFAULT_INPUT], show_default = True, multiple = True)
@click.option("-o", "--output", "output", help="Folder to output results", type = click.Path(), default = DEFAULT_OUTPUT, show_default = True)
@click.pass_context
def run(ctx: click.Context, inputs: list[Path], output: Path) -> None:
    '''Run a simulation with given inputs and output. If options aren't specified, defaults are looked-up from data/config.yaml'''
    abs_output = CONFIG_FILE.parent / output
    for input_ in inputs:
        abs_input = CONFIG_FILE.parent / input_
        click.echo(f"Loading configuration from {abs_input}, please wait a moment...")
        runner = runner_factory.create_runner(abs_input, abs_output)
        runner.run()



cli.add_command(run)
    

if __name__ == "__main__":
    print(config)
    
#%%
