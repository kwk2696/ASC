from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

