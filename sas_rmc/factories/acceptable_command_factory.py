
from abc import ABC, abstractmethod
from dataclasses import dataclass


from ..acceptance_scheme import MetropolisAcceptance, UnconditionalAcceptance
from .. import commands


@dataclass
class AcceptableCommandFactory(ABC):

    @abstractmethod
    def create_acceptable_command(self, command: commands.Command, temperature: float) -> commands.AcceptableCommand:
        pass


@dataclass
class MetropolisAcceptanceFactory(AcceptableCommandFactory):

    def create_acceptable_command(self, command: commands.Command, temperature: float) -> commands.AcceptableCommand:
        metropolis_acceptance = MetropolisAcceptance(temperature)
        return commands.AcceptableCommand(command, acceptance_scheme = metropolis_acceptance)


@dataclass
class UnconditionalAcceptanceFactory(AcceptableCommandFactory):

    def create_acceptable_command(self, command: commands.Command, temperature: float) -> commands.AcceptableCommand:
        unconditional_acceptance = UnconditionalAcceptance()
        return commands.AcceptableCommand(command, acceptance_scheme=unconditional_acceptance)



        