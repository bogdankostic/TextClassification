from multiprocessing import cpu_count

import spacy

from src.base.featurizer import BaseFeaturizer
import time

class Featurizer(BaseFeaturizer):

    def __init__(self, preprocessor, lang_model="en_core_web_sm"):

        self.spacy_model = spacy.load(lang_model, disable=["parser"])

        self.train = self._add_spacy_annotations(preprocessor.get_train_data())
        self.test = self._add_spacy_annotations(preprocessor.get_test_data())
        self.dev = self._add_spacy_annotations(preprocessor.get_dev_data())

    def add_feature(self):
        pass

    def extract_features(self):
        pass

    def _add_spacy_annotations(self, data):
        spacy_docs = self.spacy_model.pipe([sample["text"] for sample in data],
                                           n_process=cpu_count())

        for spacy_doc, sample in zip(spacy_docs, data):
            tokens = []
            lemmas = []
            pos_tags = []
            is_alpha = []
            is_stop = []
            for token in spacy_doc:
                tokens.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)
                is_alpha.append(token.is_alpha)
                is_stop.append(token.is_stop)
            sample.update(
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                is_alpha=is_alpha,
                is_stop=is_stop
            )

        return data


if __name__ == "__main__":
    from src.preprocessor.csv_preprocessor import CSVPreprocessor

    preprocessor = CSVPreprocessor("../../data/tweets.csv", delimiter=",",
                                   label_column="handle")
    featurizer = Featurizer(preprocessor)

