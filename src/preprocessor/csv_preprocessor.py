import csv
import random
import logging

from src.base.preprocessor import BasePreprocessor


class CSVPreprocessor(BasePreprocessor):

    def __init__(self, train_filename=None, test_filename=None,
                 dev_filename=None, test_split=0, dev_split=0, delimiter="\t",
                 text_column="text", label_column="label", random_state=None):

        random.seed(random_state)

        if train_filename:
            data = self._extract_data(train_filename, delimiter, text_column,
                                      label_column)
            # shuffle data if we want to use part of it as train or dev set
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

            # split data
            self.test = data[:number_of_test_samples]
            self.dev = data[number_of_test_samples:
                           number_of_test_samples+number_of_dev_samples]
            self.train = data[number_of_test_samples+number_of_dev_samples:]

            # add external test and dev data
            if test_filename:
                self.test += self._extract_data(test_filename, delimiter,
                                                text_column, label_column)
            if dev_filename:
                self.dev = self._extract_data(dev_filename, delimiter,
                                              text_column, label_column)
        else:
            self.train = []
            if test_filename:
                self.test = self._extract_data(test_filename, delimiter,
                                               text_column, label_column)
            else:
                self.test = []
            if dev_filename:
                self.dev = self._extract_data(dev_filename, delimiter,
                                              text_column, label_column)
            else:
                self.dev = []

    @classmethod
    def from_file(cls, train_filename=None, test_filename=None,
                  dev_filename=None, test_split=0, dev_split=0, delimiter="\t",
                  text_column="text", label_column="label", random_state=None):

        return cls(train_filename, test_filename, dev_filename, test_split,
                   dev_split, delimiter, text_column, label_column,
                   random_state)

    @staticmethod
    def _extract_data(filename, delimiter, text_column, label_column):
        with open(filename, "r") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            headers = next(csv_reader)
            text_col_idx = headers.index(text_column)
            label_col_idx = headers.index(label_column)
            data = [(row[text_col_idx], row[label_col_idx])
                    for row in csv_reader]

        return data
