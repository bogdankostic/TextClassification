import logging
from collections import defaultdict
import csv

from sklearn.metrics import classification_report

from text_classification.classifier.base import BaseClassifier


class ClassAverageClassifier(BaseClassifier):
    """
    A classifier that computes average feature values for each class and
    predicts the class whose average feature vector is most similar to
    the instance to predict tha class for.
    """

    def __init__(self):
        self.feature_names = []
        self.labels = []

        self._average_feature_values = dict()

    def train(self, preprocessor):
        """
        Computes the average feature vector for each class in the pre-
        processor's train set.

        :param preprocessor:
        :type preprocessor: BasePreprocessor
        :return: ClassAverageClassifier
        """
        train_set = preprocessor.get_train_data()
        if not train_set:
            logging.warning("Preprocessor's train set is empty.")

        # split train set in different classes
        split_by_label = defaultdict(list)
        for instance in train_set:
            split_by_label[instance["label"]].append(instance)
        self.labels = list(split_by_label.keys())

        # calculate average feature vector for each class
        for label in split_by_label:
            # group features values of same feature together
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
        """
        Evaluates the current model on the preprocessor's test and/or
        dev set and prints a classification report containing accuracy,
        precision, recall and F1-scores.

        :param preprocessor: Preprocessor containing dev/test data.
        :type preprocessor: BasePreprocessor
        :param evaluate_test: Whether to evaluate on the test set.
        :type evaluate_test: bool
        :param evaluate_dev: Whether to evaluate on dev set.
        :type evaluate_dev: bool
        """
        # make predictions
        self.predict(preprocessor, predict_train=False,
                     predict_test=evaluate_test, predict_dev=evaluate_dev)

        # calculate evaluation scores
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
        """
        Makes predictions for data inside preprocessor in-place, i.e.
        for each instance, a key 'prediction' containing the prediction
        is added. Instances have to be featurized before using the same
        Featurizer that was used for training instances.

        :param preprocessor: Preprocessor containing the data to make
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
        """
        Make predictions on a a list of dictionaries. Dictionaries must
        contain key 'feature_vector' consisting of the feature vector.

        :param dicts: List of dicts, where each dict represents an
            instance,
        :type dicts: List[dict]
        :return: Updated list of dictionaries.
        """
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

        return dicts

    def save_average_feature_values(self, filename, delimiter="\t"):
        """
        Saves the trained average vectors to a CSV-file.

        :param filename: File where average vectors should be saved.
        :type filename: str
        :param delimiter: Delimiter used in CSV-file.
        :type delimiter: str
        """
        with open(filename, "w") as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            csv_writer.writerow(["label"] + self.feature_names)
            for label in self._average_feature_values:
                csv_writer.writerow([label] +
                                    self._average_feature_values[label])

    @classmethod
    def load_average_feature_values(cls, filename, delimiter="\t",
                                    label_col="label"):
        """
        Loads trained average vectors from a CSV-file and instantiates
        a ClassAverageClassifier instance-

        :param filename: File where average vectors are saved
        :type filename: str
        :param delimiter: Delimiter used in CSV-file.
        :type delimiter: str
        :param label_col: Name of label column.
        :type label_col: str
        :return: ClassAverageClassifier instance.
        """
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



