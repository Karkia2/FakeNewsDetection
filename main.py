from data_loader import load_and_combine_data, explore_data
from preprocessing import preprocess_text, vectorize_text
from model import train_logistic_regression, train_naive_bayes
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Function to save evaluation results to a file
def save_evaluation_to_file(file_path, model_name, accuracy, report):
    """Saves model evaluation results to a file."""
    with open(file_path, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("\n" + "="*50 + "\n\n")

def main():
    # Load and combine the datasets
    df = load_and_combine_data('fake.csv', 'true.csv')
    explore_data(df)

    # Preprocess the text
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

    # Vectorize the text
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

    # Train Logistic Regression model
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test_tfidf))
    lr_report = classification_report(y_test, lr_model.predict(X_test_tfidf))
    print("Logistic Regression Evaluation:")
    print(lr_report)

    # Save the evaluation to a file
    save_evaluation_to_file('model_evaluation.txt', 'Logistic Regression', lr_accuracy, lr_report)

    # Train Naive Bayes model
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test_tfidf))
    nb_report = classification_report(y_test, nb_model.predict(X_test_tfidf))
    print("Naive Bayes Evaluation:")
    print(nb_report)

    # Save the evaluation to a file
    save_evaluation_to_file('model_evaluation.txt', 'Naive Bayes', nb_accuracy, nb_report)

if __name__ == "__main__":
    main()
