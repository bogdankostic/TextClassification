from multiprocessing import cpu_count
from collections import Counter, defaultdict
from statistics import mean

import spacy
from spacymoji import Emoji
import emot
import time


from src.base.featurizer import BaseFeaturizer

class Featurizer(BaseFeaturizer):

    def __init__(self, preprocessor, lang_model="en_core_web_sm",
                 normalize=True):

        self.spacy_model = spacy.load(lang_model, disable=["parser"])
        emoji_detector = Emoji(self.spacy_model)
        self.spacy_model.add_pipe(emoji_detector, first=True)
        self.normalize = normalize

        self.train = self._add_spacy_annotations(preprocessor.get_train_data())
        self.test = self._add_spacy_annotations(preprocessor.get_test_data())
        self.dev = self._add_spacy_annotations(preprocessor.get_dev_data())

        start = time.time()
        for instance in self.train:
            self.char_based_features(instance)
            self.word_based_features(instance)
        print(time.time() - start)

    def add_feature(self):
        pass

    def extract_features(self):
        pass

    #################################################
    ### Character-based features ####################
    #################################################

    def char_based_features(self, instance):
        counts = Counter(
            "alpha" if char.isalpha() else
            "upper" if char.isupper() else
            "lower" if char.islower() else
            "numeric" if char.isnumeric() else
            "whitespace" if char.isspace() else
            "comma" if char == "," else
            "dot" if char == "." else
            "exclamation" if char == "!" else
            "question" if char == "?" else
            "colon" if char == ":" else
            "semicolon" if char == ";" else
            "hyphen" if char == "-" else
            "at" if char == "@" else
            "other"
            for char in instance["text"]
        )

        # normalize counts
        if self.normalize:
            number_of_chars = len(instance["text"])
            for feature, count in counts.items():
                counts[feature] = count / number_of_chars

        instance.update({"char_features": counts})

    #################################################
    ### Word-based features #########################
    #################################################

    def word_based_features(self, instance):
        counts = defaultdict(int)

        counts["stop_words"] = sum(instance["is_stop"])
        counts["emojis"] = sum(instance["is_emoji"])
        counts["token_counts"] = len(instance["tokens"])
        counts["avg_token_len"] = mean(len(token)
                                       for token in instance["tokens"])

        if self.normalize:
            counts["stop_words"] = counts["stop_words"] / counts["token_counts"]
            counts["emojis"] = counts["emojis"] / counts["token_counts"]
        

    def _add_spacy_annotations(self, data):
        spacy_docs = self.spacy_model.pipe([sample["text"] for sample in data],
                                           n_process=cpu_count())

        for spacy_doc, sample in zip(spacy_docs, data):
            tokens = []
            lemmas = []
            pos_tags = []
            is_alpha = []
            is_stop = []
            is_emoji = []
            is_emoticon = []
            for token in spacy_doc:
                tokens.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)
                is_alpha.append(token.is_alpha)
                is_stop.append(token.is_stop)
                is_emoji.append(token._.is_emoji)
                # TODO check if token is emoticon???
                # TODO check if spelling error??
                # TODO add pos-tags
                # TODO add named entities
                # TODO add sentiment?
            sample.update(
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                is_alpha=is_alpha,
                is_stop=is_stop,
                is_emoji=is_emoji,
            )

        return data


if __name__ == "__main__":
    from src.preprocessor.csv_preprocessor import CSVPreprocessor

    preprocessor = CSVPreprocessor("../../data/tweets.csv", delimiter=",",
                                   label_column="handle")
    featurizer = Featurizer(preprocessor)

