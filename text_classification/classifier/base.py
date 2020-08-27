from abc import ABC, abstractmethod

import dill


class BaseClassifier(ABC):
    """
    Base class that all classifier classes should inherit from to ensure
    uniformity.
    """

    @abstractmethod
    def train(self, preprocessor):
        pass

    @abstractmethod
    def predict(self, preprocessor, predict_train=False, predict_test=True,
                predict_dev=False):
        pass

    @abstractmethod
    def evaluate(self, preprocessot, evaluate_test=True, evaluate_dev=False):
        pass

    def save(self, filename):
        """
        Saves current classifier instance in binary format.

        :param filename: Name of the file where the classifier should be
            saved.
        :type filename: str
        """
        with open(filename, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Loads a previously saved classifier from a binary file.

        :param filename: Name of the binary file that the classifier
            should be loaded from.
        :type filename: str

        :return: Classifier instance.
        """
        with open(filename, "rb") as file:
            classifier = dill.load(file)

        return classifier
