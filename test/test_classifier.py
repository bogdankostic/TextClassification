from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
from text_classification.featurizer.tweet_featurizer import TweetFeaturizer
from text_classification.classifier.class_average import ClassAverageClassifier


def test_train():
    # Tests whether ClassAverageClassifier extracts correct average
    # vectors from "samples/classifier_train.tsv". This file contains
    # two classes which look very different from each other. In this
    # test, we are checking whether the calculation of average features
    # is done correctly and stored properly.

    preprocessor = CSVPreprocessor(
        train_filename="samples/classifier_train.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)

    assert classifier.feature_names == \
           preprocessor.get_train_data()[0]["feature_names"]

    alpha_index = classifier.feature_names.index("alpha")
    assert classifier._average_feature_values["0"][alpha_index] == 36
    assert classifier._average_feature_values["1"][alpha_index] == 7

    upper_index = classifier.feature_names.index("upper")
    assert classifier._average_feature_values["0"][upper_index] == 0
    assert classifier._average_feature_values["1"][upper_index] == 0

    lower_index = classifier.feature_names.index("lower")
    assert classifier._average_feature_values["0"][lower_index] == 36
    assert classifier._average_feature_values["1"][lower_index] == 7

    dot_index = classifier.feature_names.index("dot")
    assert classifier._average_feature_values["0"][dot_index] == 0
    assert classifier._average_feature_values["1"][dot_index] == 4.5

    exclamation_index = classifier.feature_names.index("exclamation")
    assert classifier._average_feature_values["0"][exclamation_index] == 0
    assert classifier._average_feature_values["1"][exclamation_index] == 2

    token_count_index = classifier.feature_names.index("token_counts")
    assert classifier._average_feature_values["0"][token_count_index] == 4
    assert classifier._average_feature_values["1"][token_count_index] == 14.5


def test_predict():
    # Tests whether ClassAverageClassifier makes correct predictions.
    # for "samples/classifier_test.tsv". The classifier is trained on
    # two very differing classes and predictions are made on instances
    # which obviously persist to one of the classes.

    preprocessor = CSVPreprocessor(
        train_filename="samples/classifier_train.tsv",
        test_filename="samples/classifier_test.tsv"
    )
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)

    classifier.predict(preprocessor, predict_test=True)

    test_instances = preprocessor.get_test_data()
    assert test_instances[0]["prediction"] == "0"
    assert test_instances[1]["prediction"] == "1"


def test_save_Load():
    # Tests saving and loading of ClassAverageClassifier

    preprocessor = CSVPreprocessor(
        train_filename="samples/classifier_train.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)

    classifier.save("classifier.bin")
    loaded_classifier = ClassAverageClassifier.load("classifier.bin")

    assert classifier.feature_names == loaded_classifier.feature_names
    assert classifier.labels == loaded_classifier.labels
    assert classifier._average_feature_values == \
           loaded_classifier._average_feature_values


def test_save_load_avg_vectors():
    # Tests saving and loading of average vectors

    preprocessor = CSVPreprocessor(
        train_filename="samples/classifier_train.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)

    classifier.save_average_feature_vectors("avg_feat_vecs.tsv")
    loaded_classifier = ClassAverageClassifier.load_average_feature_vectors(
        "avg_feat_vecs.tsv")

    assert classifier.feature_names == loaded_classifier.feature_names
    assert classifier.labels == loaded_classifier.labels
    assert classifier._average_feature_values == \
           loaded_classifier._average_feature_values
