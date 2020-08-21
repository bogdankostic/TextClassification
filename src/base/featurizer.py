from abc import ABC, abstractmethod


class BaseFeaturizer(ABC):

    COARSE_POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET',
                       'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                       'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']

    @abstractmethod
    def extract_features(self):
        pass

    @abstractmethod
    def add_feature(self):
        pass
