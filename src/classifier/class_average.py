import logging
from collections import defaultdict
import csv

from sklearn.metrics import classification_report

from src.base.classifier import BaseClassifier



class ClassAverageClassifier(BaseClassifier):

    def __init__(self):
        self.feature_names = []
        self.labels = []

        self._average_feature_values = dict()

    def train(self, preprocessor):
        train_set = preprocessor.get_train_data()
        if not train_set:
            logging.warning("Preprocessor's train set is empty.")

        split_by_label = defaultdict(list)
        for instance in train_set:
            split_by_label[instance["label"]].append(instance)
        self.labels = list(split_by_label.keys())

        for label in split_by_label:
            grouped_feature_values = zip(
                *[instance["feature_vector"]
                  for instance in split_by_label[label]]
            )
            self._average_feature_values[label] = [sum(feature) /
                                                   len(split_by_label[label])
                                                   for feature
                                                   in grouped_feature_values]

        self.feature_names = train_set[0]["feature_names"]

        # evaluate on dev set
        self.evaluate(preprocessor, evaluate_test=False, evaluate_dev=True)

        return self

    def evaluate(self, preprocessor, evaluate_test=True, evaluate_dev=False):
        self.predict(preprocessor, predict_train=False,
                     predict_test=evaluate_test, predict_dev=evaluate_dev)

        if evaluate_test:
            predictions = []
            gold_labels = []
            for instance in preprocessor.get_test_data():
                predictions.append(instance["prediction"])
                gold_labels.append(instance["label"])

            print("___Evaluation metrics on test set___")
            print(classification_report(gold_labels, predictions))

        if evaluate_dev:
            predictions = []
            gold_labels = []
            for instance in preprocessor.get_dev_data():
                predictions.append(instance["prediction"])
                gold_labels.append(instance["label"])

            print("___Evaluation metrics on dev set___")
            print(classification_report(gold_labels, predictions))

    def predict(self, preprocessor, predict_train=False, predict_test=True,
                predict_dev=False):

        if predict_train:
            train_set = preprocessor.get_train_data()
            self.predict_from_dicts(train_set)

        if predict_test:
            test_set = preprocessor.get_test_data()
            self.predict_from_dicts(test_set)

        if predict_dev:
            dev_set = preprocessor.get_dev_data()
            self.predict_from_dicts(dev_set)

    def predict_from_dicts(self, dicts):
        for instance in dicts:
            if "feature_vector" in instance:
                assert (len(instance["feature_vector"]) ==
                        len(self.feature_names) and
                        instance["feature_names"] == self.feature_names), (
                    "Vectors of instances to predict and classifier "
                    "doesn't match up. Make sure to use the same Featurizer!")

                most_similar_label = self._get_most_similar_label(instance)
                instance["prediction"] = most_similar_label

            else:
                raise KeyError("Instance to predict doesn't contain feature "
                              "vector. Make sure to apply first a Featurizer!")

    def save_average_feature_values(self, filename, delimiter="\t"):
        with open(filename, "w") as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            csv_writer.writerow(["label"] + self.feature_names)
            for label in self._average_feature_values:
                csv_writer.writerow([label] +
                                    self._average_feature_values[label])

    @classmethod
    def load_average_feature_values(cls, filename, delimiter="\t",
                                    label_col="label"):
        with open(filename, "r") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            headers = next(csv_reader)
            label_col_idx = headers.index(label_col)
            headers.pop(label_col_idx)
            classifier = cls()
            classifier.feature_names = headers
            for row in csv_reader:
                label = row[label_col_idx]
                row.pop(label_col_idx)
                classifier._average_feature_values[label] = row

        return classifier

    def _get_most_similar_label(self, instance):
        similarity_values = defaultdict(float)
        instance_vector = instance["feature_vector"]

        for label in self._average_feature_values:
            average_vector = self._average_feature_values[label]
            similarity = sum([abs(instance_val - average_val)
                              for instance_val, average_val
                              in zip(instance_vector, average_vector)])
            similarity_values[label] = similarity

        most_similar_label = max(similarity_values, key=similarity_values.get)

        return most_similar_label

if __name__ == "__main__":
    from src.preprocessor.csv_preprocessor import CSVPreprocessor
    from src.featurizer.featurizer import Featurizer

    preprocessor = CSVPreprocessor("../../data/tweets.csv", delimiter=",",
                                   label_column="handle", dev_split=0.1)
    featurizer = Featurizer()
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)



