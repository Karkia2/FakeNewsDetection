import pandas as pd

def load_and_combine_data(fake_news_path, true_news_path):
    """Loads and combines fake and true news datasets."""
    # Load fake news dataset
    fake_df = pd.read_csv(fake_news_path)
    fake_df['label'] = 'fake'  # Add label for fake news

    # Load true news dataset
    true_df = pd.read_csv(true_news_path)
    true_df['label'] = 'real'  # Add label for true news

    # Combine the datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)

    return combined_df

def explore_data(df):
    """Displays basic information about the dataset."""
    print(df.head())
    print("Missing values:", df.isnull().sum())
    print("Label distribution:", df['label'].value_counts())
