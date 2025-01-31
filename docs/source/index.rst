TextClassification: Documentation
==============================================
What is it?
-----------
:code:`TextClassification` is a library, that makes text classification easily available.
It relies on three components which depend on each other and work together:

- **Preprocessor:** Reads in data and stores feature vectors and predictions
- **Featurizer:** Extracts features out of text data
- **Classifier:** Uses extracted features to train a classification model and do inference on unseen instances

Installation
------------
To make use of the TextClassification library, clone the `GitHub repository <https://github.com/bogdankostic/TextClassification>`_
and install from source:
::

   git clone https://github.com/bogdankostic/TextClassification.git
   cd TextClassificattion
   pip install -r requirements.txt
   pip install --editable .

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_preprocessors
   api_featurizers
   api_classifiers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
