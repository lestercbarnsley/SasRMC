from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol


class Loggable(Protocol):
    def get_loggable_data(self) -> dict:
        pass
    
    
@dataclass
class Converter(ABC):

    @abstractmethod
    def convert(self, command: Loggable) -> dict:
        pass


@dataclass
class ParticleConverter(Converter):
    conversion_function: Callable[[dict], dict] = None

    def convert(self, command: Loggable) -> dict:
        data = command.get_loggable_data()
        if self.conversion_function is not None:
            return self.conversion_function(data)
        return data


@dataclass
class CommandWriter(ABC):

    @abstractmethod
    def to_data(self, command: Loggable) -> dict:
        pass
    

@dataclass
class ParticleWriter(CommandWriter):

    command_converter: Converter

    def to_data(self, command: Loggable) -> dict:
        return self.command_converter.convert(command)

    @classmethod
    def standard_particle_writer(cls):
        return cls(command_converter = ParticleConverter())


if __name__ == "__main__":
    pass
