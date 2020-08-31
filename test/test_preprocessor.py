from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor


def test_read_data(sample_csv_preprocessor):
    # Test whether samples are read in completely

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

    preprocessor_data = sample_csv_preprocessor.get_train_data()

    assert all([real_instance == prep_instance for real_instance, prep_instance
                in zip(real_data, preprocessor_data)])


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

    assert all([real_instance == prep_instance for real_instance, prep_instance
                in zip(real_data, preprocessor_data)])


def test_train_split(split_csv_preprocessor):
    # Test Whether train split is made correctly
    assert len(split_csv_preprocessor.get_train_data()) == 7


def test_test_split(split_csv_preprocessor):
    # Test Whether train split is made correctly
    assert len(split_csv_preprocessor.get_test_data()) == 2


def test_dev_split(split_csv_preprocessor):
    # Test Whether train split is made correctly
    assert len(split_csv_preprocessor.get_dev_data()) == 1


def test_train_split_additional_data(split_csv_preprocessor_additional_data):
    # Test Whether train split is made correctly with additional data
    assert len(split_csv_preprocessor_additional_data.get_train_data()) == 7


def test_test_split_additional_data(split_csv_preprocessor_additional_data):
    # Test Whether train split is made correctly with additional data
    assert len(split_csv_preprocessor_additional_data.get_test_data()) == 12


def test_dev_split_additional_data(split_csv_preprocessor_additional_data):
    # Test Whether train split is made correctly with additional data
    assert len(split_csv_preprocessor_additional_data.get_dev_data()) == 11
