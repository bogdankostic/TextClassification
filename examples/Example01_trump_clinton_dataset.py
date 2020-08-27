# This example script loads the 'Hillary Clinton and Donald Trump Tweets'
# dataset and trains and evaluates a classifier on it.
# To be able to use this script, you need to download the dataset
# manually and put it in the samples folder.
# Data source: https://www.kaggle.com/benhamner/clinton-trump-tweets

from text_classification.preprocessor.csv_preprocessor import CSVPreprocessor
from text_classification.featurizer.tweet_featurizer import TweetFeaturizer
from text_classification.classifier.class_average import ClassAverageClassifier

# 1) Read in the samples. We are taking 10 % as development set and
#    20 % as test set.
preprocessor = CSVPreprocessor(
    train_filename="data/tweets.csv",
    test_split=0.2,
    dev_split=0.1,
    delimiter=",",
    label_column="handle",
    random_state=42,
)

# 2) Extract features out of the text.
featurizer = TweetFeaturizer(normalize=True)
featurizer.extract_features(preprocessor)

# 3) Train the classifier.
classifier = ClassAverageClassifier()
classifier.train(preprocessor)

# 4) Evaluate on the test set
classifier.evaluate(preprocessor)

# 5) Write predictions on the test set to csv.
preprocessor.write_csv("samples/preds.tsv", set="test")
