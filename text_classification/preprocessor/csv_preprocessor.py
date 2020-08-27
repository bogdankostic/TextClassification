import csv
import random
import logging

from text_classification.preprocessor.base import BasePreprocessor

logger = logging.getLogger(__name__)


class CSVPreprocessor(BasePreprocessor):
    """
    Preprocessor that is able to read a csv-file and do train/test/dev
    split. A preprocessor instance serves as a samples storage whose
    instances can be extended with feature vectors and predictions.
    """

    def __init__(self, train_filename=None, test_filename=None,
                 dev_filename=None, test_split=0, dev_split=0, delimiter="\t",
                 text_column="text", label_column="label", random_state=None):
        """

        :param train_filename: Train set file.
        :type train_filename: str
        :param test_filename:  Test set file.
        :type test_filename: str
        :param dev_filename: Dev set file.
        :type dev_filename: str
        :param test_split: Fraction of train set that should be used as
            test set.
        :type test_split: float
        :param dev_split: Fraction of train set that should be used as
            dev set.
        :type dev_split: float
        :param delimiter: Delimiter that is used in csv-file
        :type delimiter: str
        :param text_column: Column in csv-file containing text.
        :type text_column: str
        :param label_column: Column in csv-file containing label.
        :type label_column: str
        :param random_state: Random state for shuffling samples.
        :type random_state: int
        """

        random.seed(random_state)

        if train_filename:
            logger.info(f"Reading {train_filename}...")

            data = self._extract_data(train_filename, delimiter, text_column,
                                      label_column)
            # shuffle samples if we want to use part of it as train or dev set
            if test_split or dev_split:
                random.shuffle(data)

            # check whether test_split and dev_split are valid
            if not (0 <= test_split <= 1):
                raise ValueError(
                    f"test_split should be between 0 and 1. "
                    f"test_split is: {test_split}"
                )
            if not (0 <= dev_split <= 1):
                raise ValueError(
                    f"dev_split should be between 0 and 1. "
                    f"dev_split is: {dev_split}"
                )
            if dev_split + test_split > 1:
                raise ValueError(
                    f"Sum of test_split and dev_split must not be greater "
                    f"than 1. Sum is: {dev_split + test_split}"
                )

            # calculate number of dev and test samples
            number_of_test_samples = int(len(data) * test_split)
            number_of_dev_samples = int(len(data) * dev_split)

            # split samples into train, test and dev sets
            self.test = data[:number_of_test_samples]
            self.dev = data[number_of_test_samples:
                            number_of_test_samples+number_of_dev_samples]
            self.train = data[number_of_test_samples+number_of_dev_samples:]

            # add external test and dev samples
            if test_filename:
                logging.info(f"Reading {test_filename}...")

                self.test += self._extract_data(test_filename, delimiter,
                                                text_column, label_column)
            if dev_filename:
                logging.info(f"Reading {dev_filename}...")

                self.dev += self._extract_data(dev_filename, delimiter,
                                               text_column, label_column)
        else:
            self.train = []

            if test_filename:
                logging.info(f"Reading {test_filename}...")

                self.test = self._extract_data(test_filename, delimiter,
                                               text_column, label_column)
            else:
                self.test = []
            if dev_filename:
                logging.info(f"Reading {dev_filename}...")

                self.dev = self._extract_data(dev_filename, delimiter,
                                              text_column, label_column)
            else:
                self.dev = []

    def get_data(self):
        """
        Returns a tuple containing train, test and dev set.

        :return: Tuple with train, test and dev set.
        """
        return self.train, self.test, self.dev

    def get_train_data(self):
        """
        Returns train set.

        :return: Train set.
        """
        return self.train

    def get_test_data(self):
        """
       Returns test set.

       :return: Test set.
       :rtype: List[dict]
       """
        return self.test

    def get_dev_data(self):
        """
        Returns dev set.

        :return: Dev set.
        """
        return self.dev

    @classmethod
    def from_file(cls, train_filename=None, test_filename=None,
                  dev_filename=None, test_split=0, dev_split=0, delimiter="\t",
                  text_column="text", label_column="label", random_state=None):
        """
        Load samples from csv-files.

        :param train_filename: Train set file.
        :type train_filename: str
        :param test_filename:  Test set file.
        :type test_filename: str
        :param dev_filename: Dev set file.
        :type dev_filename: str
        :param test_split: Fraction of train set that should be used as
            test set.
        :type test_split: float
        :param dev_split: Fraction of train set that should be used as
            dev set.
        :type dev_split: float
        :param delimiter: Delimiter that is used in csv-file.
        :type delimiter: str
        :param text_column: Column in csv-file containing text.
        :type text_column: str
        :param label_column: Column in csv-file containing label.
        :type label_column: str
        :param random_state: Random state for shuffling samples.
        :type random_state: int
        :return: CSVPreprocessor instance
        """

        return cls(train_filename, test_filename, dev_filename, test_split,
                   dev_split, delimiter, text_column, label_column,
                   random_state)

    def write_csv(self, filename, delimiter="\t", set="test"):
        """
        Write samples (i.e. text, label, prediction) to a csv-file.

        :param filename: File to write the samples to.
        :type filename: str
        :param delimiter: Delimiter that is used in csv-file.
        :type delimiter: str
        :param set: Which samples set to write.
            Possible values: "train", "test", "dev"
        :type set: str
        """
        logger.info(f"Writing {set} samples to {filename}...")
        if set == "test":
            self._write_csv(filename, delimiter, self.get_test_data())
        elif set == "dev":
            self._write_csv(filename, delimiter, self.get_dev_data())
        elif set == "train":
            self._write_csv(filename, delimiter, self.get_train_data())
        else:
            raise ValueError(f"Arg set has to be one of the following values:"
                             f" 'test', 'train', 'dev'. Arg set is: {set}")

    def _write_csv(self, filename, delimiter, set):
        with open(filename, "w") as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            csv_writer.writerow(["text", "label", "prediction"])
            for instance in set:
                row = [instance.get("text", ""),
                       instance.get("label", ""),
                       instance.get("prediction", "")]
                csv_writer.writerow(row)

    def write_feature_vectors(self, filename, delimiter="\t", set="train"):
        """
        Write extracted features to a csv-file.

        :param filename: File to write the feature vectors to.
        :type filename: str
        :param delimiter: Delimiter that is used in csv-file.
        :type delimiter: str
        :param set: From which samples set to write the feature vectors.
            Possible values: "train", "test", "dev"
        :type set: str
        """
        logger.info(f"Writing {set} feature vectors to {filename}...")
        if set == "test":
            self._write_csv(filename, delimiter, self.get_test_data())
        elif set == "dev":
            self._write_csv(filename, delimiter, self.get_dev_data())
        elif set == "train":
            self._write_csv(filename, delimiter, self.get_train_data())
        else:
            raise ValueError(f"Arg set has to be one of the following values:"
                             f" 'test', 'train', 'dev'. Arg set is: {set}")

    def _write_feature_vectors(self, filename, delimiter, set):
        with open(filename, "w") as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            try:
                feat_names = set[0]["feature_names"]
                csv_writer.writerow(["text", "label", "prediction"] +
                                    feat_names)
            except IndexError:
                logger.warning("Cannot write feature vectors for empty set.")
            except KeyError:
                logger.warning("No feature vectors available. Please extract "
                               "features before wanting to write feature "
                               "vectors.")
            for instance in set:
                row = [instance.get("text", ""),
                       instance.get("label", ""),
                       instance.get("prediction", "")] + \
                      instance.get("feature_names", [0 for feat in feat_names])
                csv_writer.writerow(row)

    @staticmethod
    def _extract_data(filename, delimiter, text_column, label_column):
        with open(filename, "r") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            try:
                headers = next(csv_reader)
            except StopIteration:
                raise EOFError(f"'{filename}' is empty. Please provide a "
                               f"non-empty file or set filename to 'None' if "
                               f"you want to use an empty CSVPreprocessor.")
            try:
                text_col_idx = headers.index(text_column)
            except ValueError:
                raise ValueError(f"'{text_column}' not a column of "
                                 f"{filename}. Please provide text column"
                                 f"name.")
            try:
                label_col_idx = headers.index(label_column)
                data = [{"text": row[text_col_idx], "label": row[label_col_idx]}
                        for row in csv_reader]
            except ValueError:
                logger.warning(f"Reading data from {filename} without label, "
                               f"as column {label_column} does not exist.")
                data = [{"text": row[text_col_idx]} for row in csv_reader]

        return data
