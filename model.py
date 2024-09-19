from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    """Trains a Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model
