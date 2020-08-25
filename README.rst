TextClassification: A Python Library for simple text classification
====================================================================

What is it?
------------
The purpose of this library is to make text classification easily available.
It relies on three components which work together:

- **Preprocessor:** Reads in data and stores feature vectors and predictions
- **Featurizer:** Extracts features out of text data
- **Classifier:** Uses extracted features to train a classification model
and do inference on unseen instances

Installation
------------
To install this library, execute the following commands in your terminal:

..code-block:: bash
    git clone https://github.com/bogdankostic/TextClassification.git
    cd TextClassification
    pip install -r requirements.txt
    pip install --editable .

Usage
-----
Training a new classifier requires only three steps: 
1) Read data using a `Preprocessor` 
2) Extract features using a `Featurizer` 
3) Pass the data with extracted feature to a `Classifier`

Code example:
..code-block:: python
    from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
    from text_classification.featurizer.tweet_featurizer import TweetFeaturizer
    from text_classification.classifier.class_average import ClassAverageClassifier

    preprocessor = CSVPreprocessor(train_filename="train.tsv")

    featurizer = TweetFeaturizer()
    featurizer.extract_features(preprocessor)

    classifier = ClassAverageClassifier()
    classifier.train(preprocessor)


More detailed examples can be found in the `examples <https://github.com/bogdankostic/TextClassification/tree/master/examples>`_
directory.



 
 