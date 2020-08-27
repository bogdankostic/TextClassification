from collections import Counter

from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
from text_classification.featurizer.tweet_featurizer import TweetFeaturizer


def test_number_of_occurences_char_features():
    # Test whether featurizer extract correct number of features

    preprocessor = CSVPreprocessor(train_filename="samples/featurizer.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    data = preprocessor.get_train_data()
    char_instance = data[0]

    alpha_index = char_instance["feature_names"].index("alpha")
    assert char_instance["feature_vector"][alpha_index] == 5

    upper_index = char_instance["feature_names"].index("upper")
    assert char_instance["feature_vector"][upper_index] == 3

    lower_index = char_instance["feature_names"].index("lower")
    assert char_instance["feature_vector"][lower_index] == 2

    numeric_index = char_instance["feature_names"].index("numeric")
    assert char_instance["feature_vector"][numeric_index] == 4

    whitespace_index = char_instance["feature_names"].index("whitespace")
    assert char_instance["feature_vector"][whitespace_index] == 8

    comma_index = char_instance["feature_names"].index("comma")
    assert char_instance["feature_vector"][comma_index] == 5

    dot_index = char_instance["feature_names"].index("dot")
    assert char_instance["feature_vector"][dot_index] == 3

    exclamation_index = char_instance["feature_names"].index("exclamation")
    assert char_instance["feature_vector"][exclamation_index] == 1

    question_index = char_instance["feature_names"].index("question")
    assert char_instance["feature_vector"][question_index] == 1

    colon_index = char_instance["feature_names"].index("colon")
    assert char_instance["feature_vector"][colon_index] == 2

    semicolon_index = char_instance["feature_names"].index("semicolon")
    assert char_instance["feature_vector"][semicolon_index] == 2

    hyphen_index = char_instance["feature_names"].index("hyphen")
    assert char_instance["feature_vector"][hyphen_index] == 2

    at_index = char_instance["feature_names"].index("at")
    assert char_instance["feature_vector"][at_index] == 3

def test_number_of_occurences_word_features():
    # Test whether featurizer extract correct number of features

    preprocessor = CSVPreprocessor(train_filename="samples/featurizer.tsv")
    featurizer = TweetFeaturizer(normalize=False)
    featurizer.extract_features(preprocessor)

    data = preprocessor.get_train_data()
    word_instance = data[1]

    stopword_index = word_instance["feature_names"].index("stop_words")
    assert word_instance["feature_vector"][stopword_index] == 3

    emoji_index = word_instance["feature_names"].index("emojis")
    assert word_instance["feature_vector"][emoji_index] == 2

    token_index = word_instance["feature_names"].index("token_counts")
    assert word_instance["feature_vector"][token_index] == 10

    aux_index = word_instance["feature_names"].index("AUX")
    assert word_instance["feature_vector"][aux_index] == 2

    verb_index = word_instance["feature_names"].index("VERB")
    assert word_instance["feature_vector"][verb_index] == 1

    person_index = word_instance["feature_names"].index("PERSON")
    assert word_instance["feature_vector"][person_index] == 1

    gpe_index = word_instance["feature_names"].index("GPE")
    assert word_instance["feature_vector"][gpe_index] == 1


def test_normalization():
    # Test whether normalization is done correctly
    preprocessor = CSVPreprocessor(train_filename="samples/featurizer.tsv")
    featurizer = TweetFeaturizer(normalize=True)
    featurizer.extract_features(preprocessor)

    data = preprocessor.get_train_data()
    char_instance = data[0]
    word_instance = data[1]

    alpha_index = char_instance["feature_names"].index("alpha")
    assert abs(char_instance["feature_vector"][alpha_index] - (5 / 36)) < 0.0001

    upper_index = char_instance["feature_names"].index("upper")
    assert abs(char_instance["feature_vector"][upper_index] - (3 / 36)) < 0.0001

    lower_index = char_instance["feature_names"].index("lower")
    assert abs(char_instance["feature_vector"][lower_index] - (2 / 36)) < 0.0001

    numeric_index = char_instance["feature_names"].index("numeric")
    assert abs(char_instance["feature_vector"][numeric_index] - (4 / 36)) < 0.0001

    whitespace_index = char_instance["feature_names"].index("whitespace")
    assert abs(char_instance["feature_vector"][whitespace_index] - (8 / 36)) < 0.0001

    stopword_index = word_instance["feature_names"].index("stop_words")
    assert abs(word_instance["feature_vector"][stopword_index] - (3 / 10)) < 0.0001

    emoji_index = word_instance["feature_names"].index("emojis")
    assert abs(word_instance["feature_vector"][emoji_index] - (2 / 10)) < 0.0001

    aux_index = word_instance["feature_names"].index("AUX")
    assert abs(word_instance["feature_vector"][aux_index] - (2 / 10)) < 0.0001

    verb_index = word_instance["feature_names"].index("VERB")
    assert abs(word_instance["feature_vector"][verb_index] - (1 / 10)) < 0.0001


def test_add_feature():
    # Test functionality to add a feature

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

    data = preprocessor.get_train_data()
    char_instance = data[0]
    no_a_instance = data[2]

    a_index = char_instance["feature_names"].index("a")
    assert char_instance["feature_vector"][a_index] == 5
    assert no_a_instance["feature_vector"][a_index] == 0


def test_save_and_load():
    # Test saving and loading a featurizer

    def count_as(instance, exclude=set()):
        counts = Counter(a=0)
        for char in instance["text"]:
            if char == "a" or char == "A":
                counts["a"] += 1
        return counts

    featurizer = TweetFeaturizer(normalize=False)
    featurizer.add_feature(count_as)

    featurizer.save("featurizer.bin")
    loaded_featurizer = TweetFeaturizer.load("featurizer.bin")

    assert featurizer.spacy_model._meta["name"] == \
           loaded_featurizer.spacy_model._meta["name"]

    assert featurizer.spacy_model._meta["lang"] == \
           loaded_featurizer.spacy_model._meta["lang"]

    assert featurizer.normalize == loaded_featurizer.normalize

    assert len(featurizer.feature_functions) == \
           len(loaded_featurizer.feature_functions)
