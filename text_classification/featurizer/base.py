from abc import ABC, abstractmethod

import dill


class BaseFeaturizer(ABC):
    """
    Base class that all featurizer classes should inherit from to ensure
    uniformity.
    """

    COARSE_POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET',
                       'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                       'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

    @abstractmethod
    def extract_features(self, preprocessor, exclude=set()):
        pass

    @abstractmethod
    def add_feature(self, feature_extraction_function):
        pass

    def save(self, filename):
        """
        Saves current featurizer instance in binary format.

        :param filename: Name of the file where the featurizer should be
            saved.
        :type filename: str
        """
        with open(filename, "wb") as file:
            dill.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Loads a previously saved featurizer from a binary file.

        :param filename: Name of the binary file that the featurizer
            should be loaded from.
        :type filename: str

        :return: Classifier instance.
        """
        with open(filename, "rb") as file:
            featurizer = dill.load(file)

        return featurizer
