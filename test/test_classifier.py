from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
from text_classification.featurizer.tweet_featurizer import TweetFeaturizer
from text_classification.classifier.class_average import ClassAverageClassifier

# The following tests test whether ClassAverageClassifier extracts
# correct average vectors from "samples/classifier_train.tsv". This file
# contains two classes which look very different from each other.


def test_correct_feature_names(trained_saved_classifier):
    # Test if same feature names in preprocessor and classifier
    classifier, preprocessor, _ = trained_saved_classifier

    assert classifier.feature_names == \
           preprocessor.get_train_data()[0]["feature_names"]


def test_train_average_vector_alpha(trained_saved_classifier):
    # Test if correct average for alpha chars
    classifier, _, _ = trained_saved_classifier

    alpha_index = classifier.feature_names.index("alpha")
    assert classifier._average_feature_values["0"][alpha_index] == 36 and \
           classifier._average_feature_values["1"][alpha_index] == 7


def test_train_average_vector_upper(trained_saved_classifier):
    # Test if correct average for upper chars
    classifier, _, _ = trained_saved_classifier

    upper_index = classifier.feature_names.index("upper")
    assert classifier._average_feature_values["0"][upper_index] == 0 and \
           classifier._average_feature_values["1"][upper_index] == 0


def test_train_average_vector_lower(trained_saved_classifier):
    # Test if correct average for lower chars
    classifier, _, _ = trained_saved_classifier

    lower_index = classifier.feature_names.index("lower")
    assert classifier._average_feature_values["0"][lower_index] == 36 and \
           classifier._average_feature_values["1"][lower_index] == 7


def test_train_average_vector_dot(trained_saved_classifier):
    # Test if correct average for dots
    classifier, _, _ = trained_saved_classifier

    dot_index = classifier.feature_names.index("dot")
    assert classifier._average_feature_values["0"][dot_index] == 0 and \
           classifier._average_feature_values["1"][dot_index] == 4.5


def test_train_average_vector_exclamation(trained_saved_classifier):
    # Test if correct average for exclamation marks
    classifier, _, _ = trained_saved_classifier

    exclamation_index = classifier.feature_names.index("exclamation")
    assert classifier._average_feature_values["0"][exclamation_index] == 0 and \
           classifier._average_feature_values["1"][exclamation_index] == 2


def test_train_average_vector_token_counts(trained_saved_classifier):
    # Test if correct average for token counts
    classifier, _, _ = trained_saved_classifier

    token_count_index = classifier.feature_names.index("token_counts")
    assert classifier._average_feature_values["0"][token_count_index] == 4 and \
           classifier._average_feature_values["1"][token_count_index] == 14.5


def test_predict(trained_saved_classifier):
    # Tests whether ClassAverageClassifier makes correct predictions.
    # for "samples/classifier_test.tsv". The classifier is trained on
    # two very differing classes and predictions are made on instances
    # which obviously persist to one of the classes.
    classifier, preprocessor, _ = trained_saved_classifier

    classifier.predict(preprocessor, predict_test=True)

    test_instances = preprocessor.get_test_data()
    assert test_instances[0]["prediction"] == "0" and \
           test_instances[1]["prediction"] == "1"


def test_save_load_feature_names(trained_saved_classifier):
    # Tests correct saving and loading of feature names in
    # ClassAverageClassifier
    classifier, preprocessor, loaded_classifier = trained_saved_classifier

    assert classifier.feature_names == loaded_classifier.feature_names


def test_save_load_labels(trained_saved_classifier):
    # Tests correct saving and loading of labels in
    # ClassAverageClassifier
    classifier, preprocessor, loaded_classifier = trained_saved_classifier
    assert classifier.labels == loaded_classifier.labels


def test_save_load_avg_feature_values(trained_saved_classifier):
    # Tests correct saving and loading of avg feature values in
    # ClassAverageClassifier
    classifier, preprocessor, loaded_classifier = trained_saved_classifier
    assert classifier._average_feature_values == \
           loaded_classifier._average_feature_values


def test_save_load_avg_vectors_feature_names(trained_saved_vectors_classifier):
    # Tests correct saving and loading of feature names in
    # ClassAverageClassifier when saving feature vectors as csv
    classifier, loaded_classifier = trained_saved_vectors_classifier

    assert classifier.feature_names == loaded_classifier.feature_names


def test_save_load_avg_vectors_labels(trained_saved_vectors_classifier):
    # Tests correct saving and loading of labels in
    # ClassAverageClassifier when saving feature vectors as csv
    classifier, loaded_classifier = trained_saved_vectors_classifier

    assert classifier.labels == loaded_classifier.labels


def test_save_load_avg_vectors_avg_feature_values(
        trained_saved_vectors_classifier):
    # Tests correct saving and loading of avg feature values in
    # ClassAverageClassifier when saving feature vectors as csv
    classifier, loaded_classifier = trained_saved_vectors_classifier

    assert classifier._average_feature_values == \
           loaded_classifier._average_feature_values
