# TextClassification: A Python Library for simple text classification

## What is it?
The purpose of this library is to make text classification easily available.
It relies on three components which work together: 
- **Preprocessor:** Reads in data and stores feature vectors and predictions
- **Featurizer:** Extracts features out of text data
- **Classifier:** Uses extracted features to train a classification model
and do inference on unseen instances

The documentation where the functionality of each of the components is explained can be found [here](https://bogdankostic.github.io/TextClassification/).

## Installation
To install this library, execute the following commands in your terminal:
```bash
git clone https://github.com/bogdankostic/TextClassification.git
cd TextClassification
pip install -r requirements.txt
pip install --editable .
```

## Usage
Training a new classifier requires only three steps: 
1) Read data using a `Preprocessor` 
2) Extract features using a `Featurizer` 
3) Pass the data with extracted feature to a `Classifier`

Code example:
```python
from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
from text_classification.featurizer.tweet_featurizer import TweetFeaturizer
from text_classification.classifier.class_average import ClassAverageClassifier

preprocessor = CSVPreprocessor(train_filename="train.tsv")

featurizer = TweetFeaturizer()
featurizer.extract_features(preprocessor)

classifier = ClassAverageClassifier()
classifier.train(preprocessor)
```

## Example
This library has been built around the *Hillary Clinton and Donald Trump Tweets* dataset, which can be downloaded from
here: https://www.kaggle.com/benhamner/clinton-trump-tweets 

The example script that trains and evaluates a model on the dataset can be found in this
[here](https://github.com/bogdankostic/TextClassification/blob/master/examples/Example01_trump_clinton_dataset.py).
This example achieves an accuracy and a macro-averaged F1-score of 0.57.

## Tests
Tests can be executed using the command `pytest` from within 
the test folder `TextClassification/test`.

 
 