from abc import ABC, abstractmethod
import pickle


class BaseClassifier(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as file:
            classifier = pickle.load(file)

        return classifier
