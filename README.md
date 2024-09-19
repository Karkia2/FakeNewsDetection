FAKE NEWS DETECTION USING MACHINE LEARNING

PROJECT OVERVIEW
=================
This project builds a machine learning model to classify news articles as either real or fake. 
It applies Natural Language Processing (NLP) techniques to process textual data and trains machine learning models to predict the authenticity of news articles.

We use Logistic Regression and Naive Bayes classifiers to detect fake news, and the datasets consist of real and fake news articles. Preprocessing techniques include stopword removal and TF-IDF vectorization.


FEATURES
========
- Data Preprocessing: Stopword removal, lemmatization, and TF-IDF vectorization.
- Machine Learning Models: Logistic Regression and Naive Bayes classifiers.
- Evaluation: Accuracy, precision, recall, and F1-score metrics are used to evaluate model performance.
- Output: Results (classification reports) are saved to a file named "model_evaluation.txt".

DATASET
=======
The datasets are sourced from the "Fake and Real News Dataset" on Kaggle.

- fake.csv: Contains fake news articles.
- true.csv: Contains real news articles.

Both datasets are combined into a single dataframe with a 'label' column indicating whether the article is real or fake.


SETUP INSTRUCTIONS
==================
Prerequisites:
- Python 3.x
- Required libraries: pandas, scikit-learn, nltk, matplotlib (optional)

INSTALLATION
------------
1. Clone the repository:
   git clone https://github.com/Karkia2/FakeNewsDetection.git
   cd FakeNewsDetection

2. Install the required libraries using requirements.txt:
   pip install -r requirements.txt

3. Download NLTK data:
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')

4. Ensure the dataset files (fake.csv, true.csv) are placed in the project directory.


RUNNING THE PROJECT
===================
1. To preprocess data, train models, and evaluate results, run:
   python main.py

2. The evaluation results (accuracy, precision, recall, F1-score) will be saved in "model_evaluation.txt".



PROJECT STRUCTURE
=================
FakeNewsDetection/
  ├── fake.csv                 # Fake news dataset
  ├── true.csv                 # True news dataset
  ├── main.py                  # Main script to run the project (contains utility functions as well)
  ├── data_loader.py           # Handles loading and combining datasets
  ├── preprocessing.py         # Functions for text preprocessing
  ├── model.py                 # Functions to train and evaluate models
  ├── evaluate.py              # Functions for evaluating models
  ├── requirements.txt         # Project dependencies
  └── README.txt               # Project overview (this file)



HOW THE PROJECT WORKS
=====================
1. Data Loading: Loads two datasets (fake and real news), combines them into a dataframe, and labels them as "fake" or "real."
2. Preprocessing: Cleans text, removes stopwords, applies lemmatization, and vectorizes text using TF-IDF.
3. Model Training: Trains two models—Logistic Regression and Naive Bayes—to classify news articles.
4. Model Evaluation: Outputs accuracy, precision, recall, and F1-score, which are saved in "model_evaluation.txt".


RESULTS
=======
- Logistic Regression achieved an accuracy of X%.
- Naive Bayes achieved an accuracy of Y%.
- Full evaluation reports are available in "model_evaluation.txt".
- 
ACKNOWLEDGMENTS
===============
- Kaggle for the dataset: Fake and Real News Dataset.
- Python libraries: pandas, scikit-learn, NLTK, and matplotlib.
