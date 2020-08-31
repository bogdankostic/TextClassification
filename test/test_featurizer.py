

def test_number_of_occurences_alpha_chars(featurized_samples):
    # Test whether featurizer extracts correct number of alpha chars
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    alpha_index = char_instance["feature_names"].index("alpha")
    assert char_instance["feature_vector"][alpha_index] == 5


def test_number_of_occurences_upper_chars(featurized_samples):
    # Test whether featurizer extracts correct number of upper chars
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    upper_index = char_instance["feature_names"].index("upper")
    assert char_instance["feature_vector"][upper_index] == 3


def test_number_of_occurences_lower_chars(featurized_samples):
    # Test whether featurizer extracts correct number of lower chars
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    lower_index = char_instance["feature_names"].index("lower")
    assert char_instance["feature_vector"][lower_index] == 2


def test_number_of_occurences_numeric_chars(featurized_samples):
    # Test whether featurizer extracts correct number of numeric chars
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    numeric_index = char_instance["feature_names"].index("numeric")
    assert char_instance["feature_vector"][numeric_index] == 4


def test_number_of_occurences_whitespace_chars(featurized_samples):
    # Test whether featurizer extracts correct number of whitespace chars
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    whitespace_index = char_instance["feature_names"].index("whitespace")
    assert char_instance["feature_vector"][whitespace_index] == 8


def test_number_of_occurences_comma_chars(featurized_samples):
    # Test whether featurizer extracts correct number of commas
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    comma_index = char_instance["feature_names"].index("comma")
    assert char_instance["feature_vector"][comma_index] == 5


def test_number_of_occurences_dot_chars(featurized_samples):
    # Test whether featurizer extracts correct number of dots
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    dot_index = char_instance["feature_names"].index("dot")
    assert char_instance["feature_vector"][dot_index] == 3


def test_number_of_occurences_exclamation_chars(featurized_samples):
    # Test whether featurizer extracts correct number of exclamation marks
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    exclamation_index = char_instance["feature_names"].index("exclamation")
    assert char_instance["feature_vector"][exclamation_index] == 1


def test_number_of_occurences_question_chars(featurized_samples):
    # Test whether featurizer extracts correct number of question marks
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    question_index = char_instance["feature_names"].index("question")
    assert char_instance["feature_vector"][question_index] == 1


def test_number_of_occurences_colon_chars(featurized_samples):
    # Test whether featurizer extracts correct number of colons
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    colon_index = char_instance["feature_names"].index("colon")
    assert char_instance["feature_vector"][colon_index] == 2


def test_number_of_occurences_semicolon_chars(featurized_samples):
    # Test whether featurizer extracts correct number of semicolons
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    semicolon_index = char_instance["feature_names"].index("semicolon")
    assert char_instance["feature_vector"][semicolon_index] == 2


def test_number_of_occurences_hyphen_chars(featurized_samples):
    # Test whether featurizer extracts correct number of hyphens
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    hyphen_index = char_instance["feature_names"].index("hyphen")
    assert char_instance["feature_vector"][hyphen_index] == 2


def test_number_of_occurences_at_chars(featurized_samples):
    # Test whether featurizer extracts correct number of ats
    data = featurized_samples.get_train_data()
    char_instance = data[0]

    at_index = char_instance["feature_names"].index("at")
    assert char_instance["feature_vector"][at_index] == 3


def test_number_of_occurences_stop_words(featurized_samples):
    # Test whether featurizer extracts correct number of stop words
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    stopword_index = word_instance["feature_names"].index("stop_words")
    assert word_instance["feature_vector"][stopword_index] == 3


def test_number_of_occurences_emojis(featurized_samples):
    # Test whether featurizer extract correct number of emojis
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    emoji_index = word_instance["feature_names"].index("emojis")
    assert word_instance["feature_vector"][emoji_index] == 2


def test_number_of_tokens(featurized_samples):
    # Test whether featurizer extract correct number of tokens
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    token_index = word_instance["feature_names"].index("token_counts")
    assert word_instance["feature_vector"][token_index] == 10


def test_number_of_occurences_aux_tag(featurized_samples):
    # Test whether featurizer extract correct number of POS-tag AUX
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    aux_index = word_instance["feature_names"].index("AUX")
    assert word_instance["feature_vector"][aux_index] == 2


def test_number_of_occurences_verb_tag(featurized_samples):
    # Test whether featurizer extract correct number of POS-tag VERB
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    verb_index = word_instance["feature_names"].index("VERB")
    assert word_instance["feature_vector"][verb_index] == 1


def test_number_of_person_tag(featurized_samples):
    # Test whether featurizer extract correct number of NER-tag PERSON
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    person_index = word_instance["feature_names"].index("PERSON")
    assert word_instance["feature_vector"][person_index] == 1


def test_number_of_gpe_tag(featurized_samples):
    # Test whether featurizer extract correct number of NER-tag GPE
    data = featurized_samples.get_train_data()
    word_instance = data[1]

    gpe_index = word_instance["feature_names"].index("GPE")
    assert word_instance["feature_vector"][gpe_index] == 1


def test_number_of_occurences_alpha_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of alpha chars
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    alpha_index = char_instance["feature_names"].index("alpha")
    assert abs(char_instance["feature_vector"][alpha_index] - (5 / 36)) < 0.0001


def test_number_of_occurences_upper_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of upper chars
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    upper_index = char_instance["feature_names"].index("upper")
    assert abs(char_instance["feature_vector"][upper_index] - (3 / 36)) < 0.0001


def test_number_of_occurences_lower_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of lower chars
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    lower_index = char_instance["feature_names"].index("lower")
    assert abs(char_instance["feature_vector"][lower_index] - (2 / 36)) < 0.0001


def test_number_of_occurences_numeric_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of numeric chars
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    numeric_index = char_instance["feature_names"].index("numeric")
    assert abs(char_instance["feature_vector"][numeric_index] - (4 / 36)) \
           < 0.0001


def test_number_of_occurences_whitespace_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of whitespace chars
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    whitespace_index = char_instance["feature_names"].index("whitespace")
    assert abs(char_instance["feature_vector"][whitespace_index] - (8 / 36)) \
           < 0.0001


def test_number_of_occurences_comma_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of commas
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    comma_index = char_instance["feature_names"].index("comma")
    assert abs(char_instance["feature_vector"][comma_index] - (5 / 36)) < 0.0001


def test_number_of_occurences_dot_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of dots
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    dot_index = char_instance["feature_names"].index("dot")
    assert abs(char_instance["feature_vector"][dot_index] - (3 / 36)) < 0.0001


def test_number_of_occurences_exclamation_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of exclamation marks
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    exclamation_index = char_instance["feature_names"].index("exclamation")
    assert abs(char_instance["feature_vector"][exclamation_index] - (1 / 36)) \
           < 0.0001


def test_number_of_occurences_question_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of question marks
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    question_index = char_instance["feature_names"].index("question")
    assert abs(char_instance["feature_vector"][question_index] - (1 / 36)) \
           < 0.0001


def test_number_of_occurences_colon_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of colons
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    colon_index = char_instance["feature_names"].index("colon")
    assert abs(char_instance["feature_vector"][colon_index] - (2 / 36)) < 0.0001


def test_number_of_occurences_semicolon_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of semicolons
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    semicolon_index = char_instance["feature_names"].index("semicolon")
    assert abs(char_instance["feature_vector"][semicolon_index] - (2 / 36)) \
           < 0.0001


def test_number_of_occurences_hyphen_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of hyphens
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    hyphen_index = char_instance["feature_names"].index("hyphen")
    assert abs(char_instance["feature_vector"][hyphen_index] - (2 / 36)) \
           < 0.0001


def test_number_of_occurences_at_chars_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of ats
    data = featurized_normalized_samples.get_train_data()
    char_instance = data[0]

    at_index = char_instance["feature_names"].index("at")
    assert abs(char_instance["feature_vector"][at_index] - (3 / 36)) < 0.0001


def test_number_of_occurences_stop_words_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of stop words
    data = featurized_normalized_samples.get_train_data()
    word_instance = data[1]

    stopword_index = word_instance["feature_names"].index("stop_words")
    assert abs(word_instance["feature_vector"][stopword_index] - (3 / 10)) \
           < 0.0001


def test_number_of_occurences_emojis_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of emojis
    data = featurized_normalized_samples.get_train_data()
    word_instance = data[1]

    emoji_index = word_instance["feature_names"].index("emojis")
    assert abs(word_instance["feature_vector"][emoji_index] - (2 / 10)) < 0.0001


def test_number_of_occurences_aux_tag_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of POS-tag AUX
    data = featurized_normalized_samples.get_train_data()
    word_instance = data[1]

    aux_index = word_instance["feature_names"].index("AUX")
    assert abs(word_instance["feature_vector"][aux_index] - (2 / 10)) < 0.0001


def test_number_of_occurences_verb_tag_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of POS-tag VERB
    data = featurized_normalized_samples.get_train_data()
    word_instance = data[1]

    verb_index = word_instance["feature_names"].index("VERB")
    assert abs(word_instance["feature_vector"][verb_index] - (1 / 10)) < 0.0001


def test_number_of_person_tag_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of NER-tag PERSON
    data = featurized_normalized_samples.get_train_data()
    word_instance = data[1]

    person_index = word_instance["feature_names"].index("PERSON")
    assert abs(word_instance["feature_vector"][person_index] - (1 / 10)) \
           < 0.0001


def test_number_of_gpe_tag_normalized(
        featurized_normalized_samples):
    # Test whether featurizer normalizes correctly number of NER-tag GPE
    data = featurized_normalized_samples.get_train_data()
    word_instance = data[1]

    gpe_index = word_instance["feature_names"].index("GPE")
    assert abs(word_instance["feature_vector"][gpe_index] - (1 / 10)) < 0.0001


def test_add_feature_instance_with_feature(featurizer_added_feature):
    # Test functionality to add a feature on instance containing feature
    featurizer, preprocessor, _ = featurizer_added_feature

    data = preprocessor.get_train_data()
    char_instance = data[0]

    a_index = char_instance["feature_names"].index("a")
    assert char_instance["feature_vector"][a_index] == 5


def test_add_feature_instance_without_feature(featurizer_added_feature):
    # Test functionality to add a feature on instance not containing
    # feature
    featurizer, preprocessor, _ = featurizer_added_feature

    data = preprocessor.get_train_data()
    no_a_instance = data[2]

    a_index = no_a_instance["feature_names"].index("a")
    assert no_a_instance["feature_vector"][a_index] == 0


def test_save_and_load_spacy_model_name(featurizer_added_feature):
    # Test if same spacy model name when saving and loading a featurizer
    featurizer, preprocessor, loaded_featurizer = featurizer_added_feature

    assert featurizer.spacy_model._meta["name"] == \
           loaded_featurizer.spacy_model._meta["name"]


def test_save_and_load_spacy_model_lang(featurizer_added_feature):
    # Test if same spacy model language when saving and loading a featurizer

    featurizer, preprocessor, loaded_featurizer = featurizer_added_feature

    assert featurizer.spacy_model._meta["lang"] == \
           loaded_featurizer.spacy_model._meta["lang"]


def test_save_and_load_normalize_value(featurizer_added_feature):
    # Test if same value for normalizing when saving and loading a featurizer
    featurizer, preprocessor, loaded_featurizer = featurizer_added_feature

    assert featurizer.normalize == loaded_featurizer.normalize


def test_save_and_load_spacy_feature_funcs(featurizer_added_feature):
    # Test if same feature functions language when saving and loading a featurizer
    featurizer, preprocessor, loaded_featurizer = featurizer_added_feature

    assert all([feat.__code__ == loaded_feat.__code__
            for feat, loaded_feat
            in zip(featurizer.feature_functions,
                   loaded_featurizer.feature_functions)])
