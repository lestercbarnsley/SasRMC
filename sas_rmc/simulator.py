#%%
from dataclasses import dataclass, field
from functools import wraps
from typing import Iterator, List, Optional, Protocol, Union
import time

from .controller import Controller
from . import commands
from .acceptance_scheme import AcceptanceScheme, UnconditionalAcceptance
from .scattering_simulation import ScatteringSimulation
from .box_simulation import Box


def timeit(my_func):
    @wraps(my_func)
    def timed(*args, **kw):
    
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        
        print(f"{my_func.__name__} took {(tend - tstart)} seconds to execute")
        return output
    return timed


CommandOrAcceptableCommand = Union[commands.Command, commands.AcceptableCommand]


def decorate_command(command: CommandOrAcceptableCommand) -> commands.AcceptableCommand:
    if isinstance(command, commands.AcceptableCommand):
        return command
    if isinstance(command, commands.Command):
        return commands.AcceptableCommand(
            base_command=command,
            acceptance_scheme=UnconditionalAcceptance()
        )


class Evaluator(Protocol):
    def evaluate(self, command: CommandOrAcceptableCommand) -> bool:
        pass


@dataclass
class Simulator:
    controller: Controller
    evaluator: Evaluator

    @timeit
    def simulate(self):
        controller = self.controller
        for command in controller.ledger:
            controller.action()
            controller.compute_states()
            self.evaluator.evaluate(command)
            
@dataclass
class Callback:
    def start(self, document: dict) -> None:
        pass

    def event(self, document: dict) -> None:
        pass

    def stop(self, document: dict) -> None:
        pass

@dataclass
class Command:
    def execute(self, scattering_simulation: ScatteringSimulation | None = None) -> ScatteringSimulation:
        pass

    def get_document(self) -> dict:
        pass
@dataclass
class Controller:
    commands: list[Command]
    acceptance_scheme: list[AcceptanceScheme]

    @property
    def ledger(self) -> Iterator[tuple[Command, AcceptanceScheme]]:
        for command, acceptance in zip(self.commands, self.acceptance_scheme):
            yield command, acceptance


@dataclass
class Simulator:
    controller: Controller
    state: ScatteringSimulation
    evaluator: Evaluator
    callback: Callback

    def start(self) -> None:
        self.callback.start(self.state.get_loggable_data() | self.evaluator.get_loggable_data())

    def simulate(self) -> None:
        for command, acceptance_scheme in self.controller.ledger:
            new_state = command.execute(self.state)
            command_document = command.get_document()
            evaluation = self.evaluator.evaluate(new_state, acceptance_scheme)
            evaluation_document = self.evaluator.get_document()
            if evaluation:
                self.state = new_state
            self.callback.event(command_document | evaluation_document)

    def stop(self) -> None:
        self.callback.stop(self.state.get_loggable_data() | self.evaluator.get_loggable_data())
    

class Viewer(Protocol):
    def show_view(simulation: ScatteringSimulation, command: CommandOrAcceptableCommand, acc_scheme: AcceptanceScheme) -> None:
        pass


@dataclass
class MonteCarloEvaluator:
    simulation: ScatteringSimulation
    viewer: Optional[Viewer] = None

    def _show_view(self, command: CommandOrAcceptableCommand, acc_scheme: AcceptanceScheme) -> None:
        if self.viewer:
            self.viewer.show_view(self.simulation, command, acc_scheme)

    def evaluate(self, command: CommandOrAcceptableCommand) -> bool:
        acceptable_command = decorate_command(command)
        acceptable_command.handle_simulation(self.simulation)
        acc_scheme = acceptable_command.acceptance_scheme
        self._show_view(command, acc_scheme)
        return acc_scheme.is_acceptable()


@dataclass
class MemorizedSimulator(Simulator):
    simulation: ScatteringSimulation
    box_list: List[Box]
    state_command: commands.Command = field(init = False, default_factory=lambda : None)

    def compute_states(self) -> None:
        if self.state_command:
            self.state_command.execute()
        else:
            self.controller.compute_states()

    def simulate_command(self, controller: Controller, command: CommandOrAcceptableCommand) -> None:
        controller.action()
        self.compute_states()
        command.execute()
        acceptable = self.evaluator.evaluate(command)
        if acceptable:
            self.state_command = commands.SetSimulationState.gen_from_simulation(self.simulation.simulation_params, self.box_list)

    @timeit
    def simulate(self) -> None:
        controller = self.controller
        for command in controller.ledger:
            self.simulate_command(controller=controller, command=command)

        
if __name__ == "__main__":
    pass

   #%%