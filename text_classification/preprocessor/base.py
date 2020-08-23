from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Base class that all preprocessor classes should inherit from to
    ensure uniformity.
    """

    @classmethod
    @abstractmethod
    def from_file(self, filename):
        pass