#%%

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LogCallback:

    @abstractmethod
    def start(self, document: dict  | None = None) -> None:
        pass

    @abstractmethod
    def event(self, document: dict  | None = None) -> None:
        pass

    @abstractmethod
    def stop(self, document: dict  | None = None) -> None:
        pass


@dataclass
class NoLogCallback(LogCallback):
    def start(self, document: dict | None = None) -> None:
        pass

    def event(self, document: dict | None = None) -> None:
        pass

    def stop(self, document: dict | None = None) -> None:
        pass

@dataclass
class PrintLogCallback(LogCallback):

    def start(self, document: dict | None = None) -> None:
        print('start', document)

    def event(self, document: dict | None = None) -> None:
        print('event', document)
        
    def stop(self, document: dict | None = None) -> None:
        print('event', document)


@dataclass
class QuietLogCallback(LogCallback):
    
    def start(self, document: dict | None = None) -> None:
        pass

    def event(self, document: dict | None = None) -> None:
        if document is None:
            return None
        doc = {k : v for k, v in document.items() if k in ['Current goodness of fit', 'Cycle', 'Step', 'Acceptance']}
        if 'timestamp' in document:
            doc = doc | {'timestamp' : str(datetime.fromtimestamp(document.get('timestamp', 0)))}
        print(doc)
        

    def stop(self, document: dict | None = None) -> None:
        pass


import click

def clear_line(n=0):
    LINE_UP = f'\033[{n}A'
    click.echo(LINE_UP)

DEFAULT_KEY_LIST = ['Current goodness of fit', 'Cycle', 'Step', 'Acceptance', 'Action', 'temperature']

@dataclass
class CLILogger(LogCallback):
    current_total_lines: int = 0
    keys: list[str] = field(default_factory=lambda : DEFAULT_KEY_LIST, init = True, repr = True)
    
    def start(self, document: dict | None = None) -> None:
        HIDE_CURSOR = '\033[?25l'
        click.echo(HIDE_CURSOR + '\n')
        click.echo('Event data logger:\n')
        self.current_total_lines = 0

    def event(self, document: dict | None = None) -> None:
        if document is None:
            return None
        LINE_CLEAR = '\x1b[2K'
        clear_line(self.current_total_lines)
        doc = {k : document[k] for k in self.keys}# {k : v for k, v in document.items() if k in self.keys}
        if 'timestamp' in document:
            doc = doc | {'timestamp' : str(datetime.fromtimestamp(document.get('timestamp', 0)))}
        click.echo('\n'.join(f"{LINE_CLEAR}\t{k}: {v}" for k, v in doc.items()))
        self.current_total_lines = len(doc) + 1
        

    def stop(self, document: dict | None = None) -> None:
        UNHIDE_CURSOR = '\033[?25h'
        click.echo(UNHIDE_CURSOR + '\n')


@dataclass
class LogEventBus(LogCallback):
    log_callbacks: list[LogCallback]

    def start(self, document: dict  | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.start(document)

    def event(self, document: dict | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.event(document)

    def stop(self, document: dict | None = None) -> None:
        if document:
            timestamp = datetime.now().timestamp()
            document = document | {'timestamp' : timestamp}
        for callback in self.log_callbacks:
            callback.stop(document)



if __name__ == "__main__":
    pass

#%%