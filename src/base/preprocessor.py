from abc import ABC, abstractmethod


class BasePreprocessor(ABC):

    @classmethod
    @abstractmethod
    def read_from_file(self, filename):
        pass

    @abstractmethod
    def train_test_split(self):
        pass