from multiprocessing import cpu_count
from collections import Counter
from statistics import mean

import spacy
from spacymoji import Emoji

from src.base.featurizer import BaseFeaturizer


class Featurizer(BaseFeaturizer):

    def __init__(self, lang_model="en_core_web_sm", normalize=True):

        try:
            # initialize spacy model
            self.spacy_model = spacy.load(lang_model, disable=["parser"])
            emoji_detector = Emoji(self.spacy_model)
            self.spacy_model.add_pipe(emoji_detector, first=True)
        except OSError:
            # user inserted an unknown model name
            raise ModuleNotFoundError(
                f"Language model '{lang_model}' not installed.\n"
                f"\tTo install the model, execute: 'python -m spacy download "
                f"{lang_model}'\n"
                f"\tAvailable models can be found at: "
                f"https://spacy.io/usage/models"
            )

        self.normalize = normalize
        self.feature_functions = [
            self.char_based_features,
            self.word_based_features,
            self.pos_features,
            self.ner_features,
        ]

    def add_feature(self, feature_extraction_function):
        self.feature_functions.append(feature_extraction_function)

    def extract_features(self, preprocessor, exclude=set()):
        data_splits = preprocessor.get_data()

        for split in data_splits:
            # get annotations from spacy
            self._add_spacy_annotations(split)

            # extract features for each instance
            for instance in split:
                instance["feature_vector"] = []
                instance["feature_names"] = []
                for function in self.feature_functions:
                    count_dict = function(instance, exclude)
                    if count_dict is not None:
                        instance["feature_vector"] += list(count_dict.values())
                        instance["feature_names"] += list(count_dict.keys())

    def char_based_features(self, instance, exclude=set()):
        counts = Counter({
            "alpha": 0,
            "upper": 0,
            "lower": 0,
            "numeric": 0,
            "whitespace": 0,
            "comma": 0,
            "dot": 0,
            "exclamation": 0,
            "question": 0,
            "colon": 0,
            "semicolon": 0,
            "hyphen": 0,
            "at": 0,
        })

        for char in instance["text"]:
            if char.isalpha() and "alpha" not in exclude:
                counts["alpha"] += 1
                if char.isupper() and "upper" not in exclude:
                    counts["upper"] += 1
                elif char.islower() and "lower" not in exclude:
                    counts["lower"] += 1

            elif char.isnumeric() and "numeric" not in exclude:
                counts["numeric"] += 1
            elif char.isspace() and "whitespace" not in exclude:
                counts["whitespace"] += 1
            elif char == "," and "comma" not in exclude:
                counts["comma"] += 1
            elif char == "." and "dot" not in exclude:
                counts["dot"] += 1
            elif char == "!" and "exclamation" not in exclude:
                counts["exclamation"] += 1
            elif char == "?" and "question" not in exclude:
                counts["question"] += 1
            elif char == ":" and "colon" not in exclude:
                counts["colon"] += 1
            elif char == ";" and "semicolon" not in exclude:
                counts["semicolon"] += 1
            elif char == "-" and "hyphen" not in exclude:
                counts["hyphen"] += 1
            elif char == "@" and "at" not in exclude:
                counts["at"] += 1

        # normalize counts
        if self.normalize:
            number_of_chars = len(instance["text"])
            for feature, count in counts.items():
                counts[feature] = count / number_of_chars

        return counts

    def word_based_features(self, instance, exclude=set()):
        counts = Counter()

        counts["stop_words"] = sum(instance["is_stop"])
        counts["emojis"] = sum(instance["is_emoji"])

        token_counts = len(instance["tokens"])

        if self.normalize:
            for feature, count in counts.items():
                counts[feature] = counts[feature] / token_counts

        counts["token_counts"] = len(instance["tokens"])
        counts["avg_token_len"] = mean(len(token)
                                       for token in instance["tokens"])

        for key in exclude:
            counts.pop(key, default=None)

        return counts

    def pos_features(self, instance, exclude=set()):

        if "pos" in exclude:
            return

        counts = Counter({key: 0 for key in BaseFeaturizer.COARSE_POS_TAGS})
        for pos_tag in instance["pos_tags"]:
            counts[pos_tag] += 1

        return counts

    def ner_features(self, instance, exclude=set()):

        if "ner" in exclude:
            return

        counts = Counter({key: 0 for key in
                          self.spacy_model.pipe_labels["ner"]})
        for named_entity in instance["named_entities"]:
            counts[named_entity] += 1

        return counts

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
            for token in spacy_doc:
                tokens.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)
                is_alpha.append(token.is_alpha)
                is_stop.append(token.is_stop)
                is_emoji.append(token._.is_emoji)
                # TODO check if spelling error?? --> spacy_hunspell?

            named_entities = [span.label_ for span in spacy_doc.ents]
            sample.update(
                tokens=tokens,
                lemmas=lemmas,
                pos_tags=pos_tags,
                named_entities=named_entities,
                is_alpha=is_alpha,
                is_stop=is_stop,
                is_emoji=is_emoji,
            )


if __name__ == "__main__":
    from src.preprocessor.csv_preprocessor import CSVPreprocessor

    preprocessor = CSVPreprocessor("../../data/tweets.csv", delimiter=",",
                                   label_column="handle")
    featurizer = Featurizer()
    featurizer.extract_features(preprocessor)
    x = 1

