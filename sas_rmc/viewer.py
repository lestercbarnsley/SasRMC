from dataclasses import dataclass

from . import commands
from .acceptance_scheme import AcceptanceScheme, MetropolisAcceptance
from .scattering_simulation import ScatteringSimulation
from .simulator import CommandOrAcceptableCommand


@dataclass
class CLIViewer:

    def show_command(self, command: CommandOrAcceptableCommand):
        base_command = command.base_command if isinstance(command, commands.AcceptableCommand) else command
        print(type(base_command).__name__)
        if isinstance(base_command, commands.ParticleCommand):
            print(base_command.data, 'Particle index: ',base_command.particle_index)

    def show_simulation(self, simulation: ScatteringSimulation):
        print('rescale', simulation.simulation_params.rescale_factor)
        if simulation.simulation_params.magnetic_rescale != 1:
            print('magnetic rescale', simulation.simulation_params.magnetic_rescale)
        print('chi_squared', simulation.current_goodness_of_fit)

    def show_acceptance(self, acceptance_scheme: AcceptanceScheme):
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