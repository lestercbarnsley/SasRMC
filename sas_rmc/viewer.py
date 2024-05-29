from dataclasses import dataclass

from . import commands
from .acceptance_scheme import AcceptanceScheme, MetropolisAcceptance
from .scattering_simulation import ScatteringSimulation
from .simulator import CommandOrAcceptableCommand


@dataclass
class CLIViewer:

    def show_command(self, command: CommandOrAcceptableCommand) -> None:
        base_command = command.base_command if isinstance(command, commands.AcceptableCommand) else command
        print(type(base_command).__name__)
        if isinstance(base_command, commands.ParticleCommand):
            print(base_command.document, 'Particle index: ',base_command.particle_index)

    def show_simulation(self, simulation: ScatteringSimulation) -> None:
        rescale_factor = simulation.simulation_params.params[0].value
        magnetic_rescale = simulation.simulation_params.params[1].value # This is the perfect example of poor coupling!!!
        print('rescale', rescale_factor)
        if magnetic_rescale != 1:
            print('magnetic rescale', magnetic_rescale)
        print('chi_squared', simulation.current_goodness_of_fit)

    def show_acceptance(self, acceptance_scheme: AcceptanceScheme) -> None:
        print('step accepted' if acceptance_scheme.is_acceptable() else 'step rejected')
        if isinstance(acceptance_scheme, MetropolisAcceptance):
            print('temperature',acceptance_scheme.temperature)

    def show_view(self, simulation: ScatteringSimulation = None, command: CommandOrAcceptableCommand = None, acceptance_scheme: AcceptanceScheme = None):
        '''This is just printing out a bunch of diagnostics'''
        if command:
            self.show_command(command)
        if simulation:
            self.show_simulation(simulation)
        if acceptance_scheme:
            self.show_acceptance(acceptance_scheme)