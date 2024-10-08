from dataclasses import dataclass
from typing import Iterator

from sas_rmc.commands import Command
from sas_rmc.acceptance_scheme import AcceptanceScheme


@dataclass
class ControlStep:
    command: Command
    acceptance_scheme: AcceptanceScheme


@dataclass
class Controller:
    ledger: list[ControlStep]
    '''commands: list[Command]
    acceptance_scheme: list[AcceptanceScheme]

    @property
    def ledger(self) -> Iterator[tuple[Command, AcceptanceScheme]]:
        for command, acceptance in zip(self.commands, self.acceptance_scheme):
            yield command, acceptance'''



    