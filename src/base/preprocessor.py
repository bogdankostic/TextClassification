from abc import ABC, abstractmethod


class BasePreprocessor(ABC):

    @classmethod
    @abstractmethod
    def from_file(self, filename):
        pass