from collections import Counter

import pytest

from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
from text_classification.featurizer.tweet_featurizer import TweetFeaturizer
from text_classification.classifier.class_average import ClassAverageClassifier


@pytest.fixture
def sample_csv_preprocessor():
    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data.tsv"
    )

    return preprocessor


@pytest.fixture
def split_csv_preprocessor():
    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data.tsv",
        test_split=0.2,
        dev_split=0.1
    )

    return preprocessor


@pytest.fixture
def split_csv_preprocessor_additional_data():
    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data.tsv",
        test_filename="samples/sample_data.tsv",
        dev_filename="samples/sample_data.tsv",
        test_split=0.2,
        dev_split=0.1
    )

    return preprocessor


@pytest.fixture
def featurized_samples():
    preprocessor = CSVPreprocessor(train_filename="samples/featurizer.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    return preprocessor


@pytest.fixture
def featurized_normalized_samples():
    preprocessor = CSVPreprocessor(train_filename="samples/featurizer.tsv")
    featurizer = TweetFeaturizer(normalize=True)
    featurizer.extract_features(preprocessor)

    return preprocessor


@pytest.fixture
def featurizer_added_feature():

    def count_as(instance, exclude=set()):
        counts = Counter(a=0)
        for char in instance["text"]:
            if char == "a" or char == "A":
                counts["a"] += 1
        return counts

    preprocessor = CSVPreprocessor(train_filename="samples/featurizer.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.add_feature(count_as)
    featurizer.extract_features(preprocessor)

    featurizer.save("featurizer.bin")
    loaded_featurizer = TweetFeaturizer.load("featurizer.bin")

    return featurizer, preprocessor, loaded_featurizer


@pytest.fixture
def trained_saved_classifier():
    preprocessor = CSVPreprocessor(
        train_filename="samples/classifier_train.tsv",
        test_filename="samples/classifier_test.tsv"
    )
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)

    classifier.save("classifier.bin")
    loaded_classifier = ClassAverageClassifier.load("classifier.bin")

    return classifier, preprocessor, loaded_classifier


@pytest.fixture
def trained_saved_vectors_classifier():
    preprocessor = CSVPreprocessor(
        train_filename="samples/classifier_train.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)

    classifier.save_average_feature_vectors("avg_feat_vecs.tsv")
    loaded_classifier = ClassAverageClassifier.load_average_feature_vectors(
        "avg_feat_vecs.tsv")

    return classifier, loaded_classifier
