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