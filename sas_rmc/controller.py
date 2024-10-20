from dataclasses import dataclass

from sas_rmc import Command
from sas_rmc.acceptance_scheme import AcceptanceScheme


@dataclass
class ControlStep:
    command: Command
    acceptance_scheme: AcceptanceScheme


@dataclass
class Controller:
    ledger: list[ControlStep]
    



    