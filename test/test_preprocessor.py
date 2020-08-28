from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor


def test_read_data():
    # Test whether samples is read in completely

    real_data = [
        {"text": "text 1", "label": "0"},
        {"text": "text 2", "label": "1"},
        {"text": "text 3", "label": "0"},
        {"text": "text 4", "label": "1"},
        {"text": "text 5", "label": "0"},
        {"text": "text 6", "label": "1"},
        {"text": "text 7", "label": "0"},
        {"text": "text 8", "label": "1"},
        {"text": "text 9", "label": "0"},
        {"text": "text 10", "label": "1"},
    ]

    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data.tsv"
    )

    preprocessor_data = preprocessor.get_train_data()

    for real_instance, pre_instance in zip(real_data, preprocessor_data):
        assert real_instance == pre_instance


def test_different_column_names():
    # Test whether different column names can be passed
    # Test whether samples is read in completely

    real_data = [
        {"text": "text 1", "label": "0"},
        {"text": "text 2", "label": "1"},
        {"text": "text 3", "label": "0"},
        {"text": "text 4", "label": "1"},
        {"text": "text 5", "label": "0"},
        {"text": "text 6", "label": "1"},
        {"text": "text 7", "label": "0"},
        {"text": "text 8", "label": "1"},
        {"text": "text 9", "label": "0"},
        {"text": "text 10", "label": "1"},
    ]

    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data_different_columns.tsv",
        text_column="string",
        label_column="class"
    )

    preprocessor_data = preprocessor.get_train_data()

    for real_instance, pre_instance in zip(real_data, preprocessor_data):
        assert real_instance == pre_instance


def test_splits():
    # Test whether test and dev splits are made correctly

    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data.tsv",
        test_split=0.2,
        dev_split=0.1
    )

    assert len(preprocessor.get_train_data()) == 7
    assert len(preprocessor.get_test_data()) == 2
    assert len(preprocessor.get_dev_data()) == 1


def test_splits_with_additional_data():
    # Test whether additional test and dev samples is handled correctly

    preprocessor = CSVPreprocessor(
        train_filename="samples/sample_data.tsv",
        test_filename="samples/sample_data.tsv",
        dev_filename="samples/sample_data.tsv",
        test_split=0.2,
        dev_split=0.1
    )

    assert len(preprocessor.get_train_data()) == 7
    assert len(preprocessor.get_test_data()) == 12
    assert len(preprocessor.get_dev_data()) == 11
