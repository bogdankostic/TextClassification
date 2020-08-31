from abc import ABC, abstractmethod

import dill


class BaseClassifier(ABC):
    """
    Base class that all classifier classes should inherit from to ensure
    uniformity.
    """

    @abstractmethod
    def train(self, preprocessor):
        """
        Should train the classifier on the preprocessor's train set.

        :param preprocessor: Preprocessor instance that contains the
            train set to train the classifier on.
        :type preprocessor: BasePreprocessor
        :return: BaseClassifier
        """
        pass

    @abstractmethod
    def predict(self, preprocessor, predict_train=False, predict_test=True,
                predict_dev=False):
        """
        Should add the field :code:`prediction` to the preprocessor's
        instances containing the predicted label.

        :param preprocessor: Preprocessor containing the samples to make
            predictions on.
        :type preprocessor: BasePreprocessor
        :param predict_train: Whether to make predictions on the
            train set.
        :type predict_train: bool
        :param predict_test: Whether to make predictions on the test set.
        :type predict_test: bool
        :param predict_dev: Whether to make predictions on the dev set.
        :type predict_dev: bool
        """
        pass

    @abstractmethod
    def evaluate(self, preprocessor, evaluate_test=True, evaluate_dev=False):
        """
        Should make predictions on the preprocessor's train and/or dev
        set and print out evaluation metrics.

        :param preprocessor: Preprocessor containing dev/test samples.
        :type preprocessor: BasePreprocessor
        :param evaluate_test: Whether to evaluate on the test set.
        :type evaluate_test: bool
        :param evaluate_dev: Whether to evaluate on dev set.
        :type evaluate_dev: bool
        """
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
