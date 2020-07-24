from abc import ABC, abstractmethod


class BaseFeaturizer(ABC):

    @abstractmethod
    def extract_features(self):
        pass

    @abstractmethod
    def add_feature(self):
        pass
