#%%
from pathlib import Path

import click
import yaml

from sas_rmc.factories import runner_factory

CONFIG_FILE = Path(__file__).parent.parent / Path("data") / Path("config.yaml")

with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

#print(CONFIG_FILE)
DEFAULT_INPUT = Path(config.get('input_config_source'))
DEFAULT_OUTPUT =  Path(config.get('output_folder'))


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx: click.Context):
    '''SasRMC is a Python library for numerical modelling of small-angle scattering data.'''
    pass


@click.command()
@click.option("-i", "--input", "inputs", help="List of Excel files containing small-angle scattering data and simulation configurations to run", type = list, default = [DEFAULT_INPUT], show_default = True, multiple = True)
@click.option("-o", "--output", "output", help="Folder to output results", type = click.Path(), default = DEFAULT_OUTPUT, show_default = True)
@click.pass_context
def run(ctx: click.Context, inputs: list[Path], output: Path) -> None:
    abs_output = CONFIG_FILE.parent / output
    print(inputs)
    for input_ in inputs:
        abs_input = CONFIG_FILE.parent / input_
        click.echo(f"Loading configuration from {abs_input}, please wait a moment...")
        click.echo(f'{abs_input=}')
        click.echo(f'{abs_output=}')
        runner = runner_factory.create_runner(abs_input, abs_output)
        #runner.run()

cli.add_command(run)
    

if __name__ == "__main__":
    print(config)
    
#%%
