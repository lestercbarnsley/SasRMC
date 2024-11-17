#%%
from pathlib import Path

import click
import yaml

from sas_rmc.factories import runner_factory

CONFIG_FILE = Path(__file__).parent.parent / Path("data") / Path("config.yaml")

with open(CONFIG_FILE) as f:
    current_config = yaml.load(f, Loader=yaml.FullLoader)

DEFAULT_INPUT = Path(current_config.get('input_config_source'))
DEFAULT_OUTPUT = Path(current_config.get('output_folder'))
DEFAULT_TEMPLATE_SOURCE = current_config.get('template_source')


@click.group()
@click.version_option()
@click.pass_context
def cli(ctx: click.Context):
    '''SasRMC is a Python library for numerical modelling of small-angle scattering data.'''
    pass

@click.command()
@click.option("-i", "--input", "inputs", help="Excel file(s) containing small-angle scattering data and simulation configurations to run.", type = click.Path(), default = [DEFAULT_INPUT], show_default = True, multiple = True)
@click.option("-o", "--output", "output", help="Folder to output results.", type = click.Path(), default = DEFAULT_OUTPUT, show_default = True)
@click.pass_context
def run(ctx: click.Context, inputs: list[Path], output: Path) -> None:
    '''Run a simulation with given inputs and output. If options aren't specified, defaults are looked-up from data/config.yaml'''
    abs_output = CONFIG_FILE.parent / output
    for input_ in inputs:
        abs_input = Path.cwd() / input_
        click.echo(f"Loading configuration from {abs_input}, please wait a moment...")
        runner = runner_factory.create_runner(abs_input, abs_output)
        runner.run()


@click.group()
def config():
    """Configuration options."""
    pass

def show_settings(settings: dict):
    for setting in settings:
        click.echo(f"\t{setting}: {settings[setting]}")

@config.command()
def show():
    """Show the current configuration."""
    if not CONFIG_FILE.exists():
        click.echo("No configuration found.")
        return

    show_settings(current_config)


@config.command()
@click.option("-i", "--input", "input", type=click.Path(), help="Update the default .xlsx file to run when otherwise unspecified.", default = DEFAULT_INPUT, show_default = True)
@click.option("-o", "--output", "output", type = click.Path(), help="Update the default folder where the output is written.", default = DEFAULT_OUTPUT, show_default = True)
@click.option("-t", "--template", "template", type = click.STRING, help="Update the repository for downloading template files.", default = DEFAULT_TEMPLATE_SOURCE, show_default = True)
@click.confirmation_option(prompt='Are you sure you want to update the config?')
def update(input: Path, output: Path, template: str):
    """Update the current configuration."""
    if input != DEFAULT_INPUT:
        current_config["input_config_source"] = str(input)
    if output != DEFAULT_OUTPUT:
        current_config["output_folder"] = str(output)
    if template != DEFAULT_TEMPLATE_SOURCE:
        current_config["template_source"] = str(output)

    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(current_config, f, Dumper = yaml.Dumper)

    click.echo(f"Config updated.")

    show_settings(current_config)


@click.group()
@click.argument('template-type', type=click.Choice(['coreshell', 'dumbbell'], case_sensitive=False))
@click.option("-o", "--output", "output", help="Folder to output results.", type = click.Path(), default = DEFAULT_OUTPUT, show_default = True)
def create(template_type):
    """Create a template file by downloading it from the repository."""
    if template_type == 'coreshell':
        pass

#@


cli.add_command(run)
cli.add_command(config)
    

if __name__ == "__main__":
    print(config)
    
#%%
