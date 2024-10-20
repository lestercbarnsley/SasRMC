#%%

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt, patches, figure, colors as mcolors
import numpy as np
import pandas as pd

from sas_rmc.vector import Vector
from sas_rmc.constants import PI


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
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    fig.set_size_inches(4,4)
    d_0, d_1 = 14000, 14000
    ax.set_xlim(-d_0 / 2, +d_0 / 2)
    ax.set_ylim(-d_1 / 2, +d_1 / 2)

    ax.set_aspect("equal")

    ax.set_xlabel(r'X (Angstrom)',fontsize =  14)
    ax.set_ylabel(r'Y (Angstrom)',fontsize =  14)

    patch_list = [
            patches.Circle(
                xy = (0, 0),
                radius=120,
                ec = None,
                fc = 'black'
            ),
            patches.Circle(
                xy = (0, 0),
                radius=100,
                ec = None,
                fc = 'blue'
            )
        ]

    patch_list_2 = [
            patches.Circle(
                xy = (0,120 + 120),
                radius=120,
                ec = None,
                fc = 'black'
            ),
            patches.Circle(
                xy = (0,120 + 120),
                radius=100,
                ec = None,
                fc = 'blue'
            )
        ]

    for patch in patch_list + patch_list_2:
        #patch.set_snap(False)
        ax.add_patch(patch)



    #ax.set_box_aspect(d_1 / d_0)

    #fig.tight_layout()
    #print(fig.)
    fig.show()
    fig.savefig('test.pdf')

#%%