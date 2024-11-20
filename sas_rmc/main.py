#%%
from pathlib import Path

import click
import yaml
import requests

from sas_rmc.factories import runner_factory


CONFIG_FILE = Path(__file__).parent / Path("data") / Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        click.echo("No configuration found.")
        raise FileNotFoundError("No configuration found.")
    with open(CONFIG_FILE) as f:
        current_config = yaml.load(f, Loader=yaml.FullLoader)
    return current_config

def save_config(current_config: dict) -> None:
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(current_config, f, Dumper = yaml.Dumper)


current_config = load_config()

DEFAULT_INPUT = Path(current_config.get('input_config_source', ''))
DEFAULT_OUTPUT = Path(current_config.get('output_folder', ''))
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
    abs_output = output if output.exists() else Path.cwd() / output
    for input_ in inputs:
        abs_input = Path.cwd() / input_
        click.echo(f"Loading configuration from {abs_input}, please wait a moment...")
        runner = runner_factory.create_runner(abs_input, abs_output)
        runner.run()

@click.group()
def config():
    """Configuration options."""
    pass

def show_settings(current_config: dict):
    for setting in current_config:
        click.echo(f"\t{setting}: {current_config[setting]}")

@config.command()
def show():
    """Show the current configuration."""
    current_config = load_config()
    if not current_config:
        return None
    show_settings(current_config)


@config.command()
@click.option("-i", "--input", "input", type=click.Path(), help="Update the default .xlsx file to run when otherwise unspecified.", default = DEFAULT_INPUT, show_default = True)
@click.option("-o", "--output", "output", type = click.Path(), help="Update the default folder where the output is written.", default = DEFAULT_OUTPUT, show_default = True)
@click.option("-t", "--template", "template", type = click.STRING, help="Update the repository for downloading template files.", default = DEFAULT_TEMPLATE_SOURCE, show_default = False)
@click.confirmation_option(prompt='Are you sure you want to update the config?')
def update(input: Path, output: Path, template: str):
    """Update the current configuration."""
    current_config = load_config()
    if current_config is None:
        return None
    
    if input != DEFAULT_INPUT:
        current_config["input_config_source"] = str(input)
    if output != DEFAULT_OUTPUT:
        current_config["output_folder"] = str(output)
    if template != DEFAULT_TEMPLATE_SOURCE:
        current_config["template_source"] = str(output)
    
    click.echo(f"Config updated.")

    show_settings(current_config)


@cli.command()
@click.argument('template-type', type=click.Choice(['coreshell-1d', 'coreshell-2d','particle-1d', 'particle-2d', 'cylinder-1d', 'dumbbell-2d'], case_sensitive=False))
@click.option("-o", "--output", "output", help="Folder where the template will be saved.", type = click.Path(), default = DEFAULT_OUTPUT, show_default = True)
def create(template_type: str, output: Path):
    """Create a template file by downloading it from the repository. Specify a template type from one of the available choices."""
    
    abs_output = output if output.exists() else Path.cwd() / output
    click.echo(f"Downloading {template_type}")
    template_source = load_config().get('template_source' , '')
    if template_type == 'coreshell-2d':
        url = template_source + '/data/CoreShell_Simulation_Input.xlsx'
        response = requests.get(url)
        save_destination = abs_output / Path("CoreShell_Simulation_Input.xlsx")
        if save_destination.exists():
            click.confirm(f'File already exists at {save_destination.absolute()}. Are you sure you want to continue?', abort=True)
        with open(save_destination, mode = 'wb') as f:
            f.write(response.content)
        click.echo(f"Template {template_type} has been saved to {save_destination.absolute()}")

#@


cli.add_command(run)
cli.add_command(config)
#cli.add_command(create)
    

if __name__ == "__main__":
    pass
    
#%%
