from dataclasses import dataclass, field
from typing import List, Protocol


class CommandProtocol(Protocol):
    def execute(self) -> None:
        pass


@dataclass
class Controller:
    ledger: List[CommandProtocol] = field(default_factory=list)
    _current: int = 0

    @property
    def completed_commands(self) -> List[CommandProtocol]:
        return self.ledger[: self._current]

    def action(self) -> None:
        if self._current < len(self.ledger):
            self._current += 1

    def compute_states(self) -> None:
        for command in self.completed_commands:
            command.execute()

    def add_command(self, command: CommandProtocol) -> None:
        self.ledger.append(command)




    