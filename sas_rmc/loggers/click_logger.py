from dataclasses import dataclass, field
from datetime import datetime

import click
from colorama import just_fix_windows_console

from sas_rmc.loggers import LogCallback


def clear_line(n=0):
    LINE_UP = f'\033[{n}A'
    click.echo(LINE_UP)

DEFAULT_KEY_LIST = ['Current goodness of fit', 'Cycle', 'Step', 'Acceptance', 'Action', 'temperature']


@dataclass
class CLIckLogger(LogCallback):
    current_total_lines: int = 0
    keys: list[str] = field(default_factory=lambda : DEFAULT_KEY_LIST, init = True, repr = True)

    def __post_init__(self):
        just_fix_windows_console()
    
    def start(self, document: dict | None = None) -> None:
        HIDE_CURSOR = '\033[?25l'
        click.echo(HIDE_CURSOR) 
        click.echo('Event data logger:', nl=True)
        self.current_total_lines = 0

    def event(self, document: dict | None = None) -> None:
        if document is None:
            return None
        LINE_CLEAR = '\x1b[2K'
        clear_line(self.current_total_lines)
        doc = {k : document[k] for k in self.keys}
        if 'timestamp' in document:
            doc = doc | {'timestamp' : str(datetime.fromtimestamp(document.get('timestamp', 0)))}
        click.echo('\n'.join(f"{LINE_CLEAR}\t{k}: {v}" for k, v in doc.items()))
        self.current_total_lines = len(doc) + 1
        
    def stop(self, document: dict | None = None) -> None:
        UNHIDE_CURSOR = '\033[?25h'
        click.echo(UNHIDE_CURSOR, nl = True)


if __name__ == "__main__":
    pass

