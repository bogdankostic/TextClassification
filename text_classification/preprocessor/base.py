from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Base class that all preprocessor classes should inherit from to
    ensure uniformity. Classes inheriting from BasePreprocessor should
    consist of at least these three instance variables: train, test and
    dev. Each of these variables should contain the data split
    corresponding to its name. Each data split should be a list of
    dictionaries, where each dictionary represents one instance and
    contains the fields :code:`text` and :code:`label` holding the
    instance's corresponding value.
    ::
        self.train, self.test, self.dev = [
            {
                "text": instance_1 text,
                "label": instance_1 label
            },
            {
                "text": instance_2 text,
                "label": instance_2 label
            },
            ...
        ]

    """

    @classmethod
    @abstractmethod
    def from_file(cls, filename):
        pass

    @abstractmethod
    def write_csv(self, filename, delimiter):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def get_dev_data(self):
        pass
